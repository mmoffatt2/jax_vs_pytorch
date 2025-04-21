import os
import json
import time
import torch
from transformers import AutoModel, AutoTokenizer, FlaxAutoModel
import transformers
import shutil
import tempfile
import torch_xla.core.xla_model as xm  # for PyTorch XLA
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm     # pip install tqdm

# from torch._subclasses.fake_tensor import FakeTensorMode
# import torch._dynamo as dynamo
# print("suppress_errors:", dynamo.config.suppress_errors)
# fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
# print("allow_non_fake_inputs (class):", FakeTensorMode.allow_non_fake_inputs)
# Create a mode that will auto‑convert real Tensors/scalars to FakeTensors

# Run beforehand:
# export GPU_NUM_DEVICES=1
# export XLA_USE_CUDA=1
# export PJRT_DEVICE=cuda
# export LD_LIBRARY_PATH=/opt/conda/envs/xla-env/lib:$LD_LIBRARY_PATH

# export TORCH_LOGS="+dynamo,+inductor"
# export TORCHDYNAMO_VERBOSE=1

def append_results(new_results, output_file="paper_model_results_sentencepiece.json"):
    # Check if the file exists and load its data, else start with an empty list.
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # If the file is empty or invalid, start fresh.
            data = []
    else:
        data = []
    
    # Append new results. Depending on your design, you can append the entire results
    # or just the current model's result. Here, we're assuming new_results is a dict.
    data.append(new_results)
    
    # Write back the updated list to the file.
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Results successfully appended to {output_file}")

# Define helper functions for creating dummy inputs.
# Here, we simply tokenize a fixed string; in practice you might generate random inputs
def get_dummy_inputs(tokenizer, device=None):
    # Create inputs for a transformer (adjust as needed)
    dummy_text = "This is a dummy input for benchmarking."
    inputs = tokenizer(dummy_text, return_tensors="pt")
    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

# def benchmark_across_iterations(model, inputs, backend="inductor"):
#     results = []
#     try:
#         compile_start = time.time()
#         compiled = torch.compile(model, backend=backend)
#         compile_end = time.time()
#         compile_time = compile_end - compile_start
#     except Exception as e:
#         return f"Failed compilation due to {e}"

#     # Run a single long inference and measure time after each iteration
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     timestamps = []
#     try:
#         start = time.time()
#         for i in range(10000):
#             _ = compiled(**inputs)
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             timestamps.append(time.time() - start)
#     except Exception as e:
#         return [{"iterations": i + 1, "compile_time_s": compile_time, "error": str(e)} for i in 100]

#     for i, inf_time in enumerate(timestamps):
#         if i % 100 == 0:
#             results.append({
#                 "iterations": i,
#                 "compile_time_s": compile_time,
#                 "inference_time_s": inf_time,
#                 "total_time_s": compile_time + inf_time
#             })

#     return results

# def benchmark_eager_across_iterations(model, inputs):
#     results = []
#     model.eval()
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     timestamps = []
#     try:
#         for i in range(10000):
#             if i == 0:
#                 start = time.time()
#             _ = model(**inputs)
#             torch.cuda.synchronize() if torch.cuda.is_available() else None
#             timestamps.append(time.time() - start)
#     except Exception as e:
#         return [{"iterations": i + 1, "error": str(e)} for i in 100]

#     for i, inf_time in enumerate(timestamps):
#         if i % 100 == 0:
#             results.append({
#                 "iterations": i,
#                 "inference_time_s": inf_time,
#             })

#     return results

def benchmark_across_iterations(model, inputs, backend="inductor"):
    results = []
    try:
        compile_start = time.time()
        compiled = torch.compile(model, backend=backend)
        compile_end = time.time()
        compile_time = compile_end - compile_start
    except Exception as e:
        return f"Failed compilation due to {e}"

    compiled.train()
    optimizer = torch.optim.SGD(compiled.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Create dummy target based on model output
    with torch.no_grad():
        sample_output = compiled(**inputs)[0]
    dummy_target = torch.randn_like(sample_output)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    timestamps = []

    try:
        start = time.time()
        for i in range(3000):
            optimizer.zero_grad()
            output = compiled(**inputs)[0]
            loss = loss_fn(output, dummy_target)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            timestamps.append(time.time() - start)
    except Exception as e:
        return [{"iterations": i + 1, "compile_time_s": compile_time, "error": str(e)} for i in range(100)]

    for i, t in enumerate(timestamps):
        if i % 100 == 0:
            results.append({
                "iterations": i,
                "compile_time_s": compile_time,
                "training_time_s": t,
                "total_time_s": compile_time + t
            })

    return results

def benchmark_eager_across_iterations(model, inputs):
    results = []
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        sample_output = model(**inputs)[0]
    dummy_target = torch.randn_like(sample_output)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    timestamps = []

    try:
        for i in range(3000):
            if i == 0:
                start = time.time()
            optimizer.zero_grad()
            output = model(**inputs)[0]
            loss = loss_fn(output, dummy_target)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            timestamps.append(time.time() - start)
    except Exception as e:
        return [{"iterations": i + 1, "error": str(e)} for i in range(100)]

    for i, t in enumerate(timestamps):
        if i % 100 == 0:
            results.append({
                "iterations": i,
                "training_time_s": t,
            })

    return results


def run_benchmarks():
    # ─── List of GPT‑2 variants ────────────────────────────────────────────────
    gpt2_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

    all_results = {}

    for model_id in tqdm(gpt2_models, desc="GPT-2 Variants", unit="model"):
        all_results[model_id] = {}
        tmp_cache = tempfile.mkdtemp()
        try:
            # try:
            # 1) Load tokenizer (fast, fallback to slow)
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=tmp_cache)
            except ValueError:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, cache_dir=tmp_cache, use_fast=False
                )
            dummy_inputs = get_dummy_inputs(tokenizer)

            # 2) Load PyTorch model
            pt_model = AutoModel.from_pretrained(model_id, cache_dir=tmp_cache).eval()

            # 3) If encoder‑decoder (not the case for GPT‑2), add shift logic
            if getattr(pt_model.config, "is_encoder_decoder", False):
                if pt_model.config.model_type == "t5":
                    from transformers.models.t5.modeling_t5 import shift_tokens_right
                else:
                    from transformers.models.bart.modeling_bart import shift_tokens_right

                decoder_input_ids = shift_tokens_right(
                    dummy_inputs["input_ids"],
                    pt_model.config.pad_token_id,
                    pt_model.config.decoder_start_token_id
                )
                dummy_inputs["decoder_input_ids"] = decoder_input_ids

            # 4) Eager benchmark
            eager_results = benchmark_eager_across_iterations(pt_model, dummy_inputs)
            all_results[model_id]["eager"] = eager_results

            # 5) torch.compile backends
            # backends = ["inductor", "cudagraphs", "onnxrt", "openxla", "tvm"]
            backends = ["inductor"]
            for backend in tqdm(backends,
                                desc=f"{model_id}… backends",
                                leave=False,
                                unit="backend"):
                key = f"benchmark_iterations_{backend}"
                results = benchmark_across_iterations(
                    pt_model, dummy_inputs, backend=backend
                )
                all_results[model_id][key] = results

            # except Exception as e:
            #     # record any per‑model error
            #     all_results[model_id] = {"error": repr(e)}

        finally:
            shutil.rmtree(tmp_cache)

    return all_results


if __name__ == "__main__":
    results = run_benchmarks()
    print("Done:", results)
    with open("gpt2_ablation_training.json", "w") as f:
        json.dump(results, f, indent=4)