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

# Inference benchmark for PyTorch with torch.compile
def benchmark_inference_torch_compile(model, inputs, num_iters=100):
    # Warm-up
    for _ in range(10):
        _ = model(**inputs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(num_iters):
        _ = model(**inputs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    return (end - start) / num_iters

# Training benchmark for PyTorch with torch.compile
def benchmark_training_torch_compile(model, inputs, num_iters=100):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    # Create a dummy target tensor (using model output shape as reference)
    with torch.no_grad():
        sample_output = model(**inputs)[0]
    dummy_target = torch.randn_like(sample_output)
    # Warm-up
    for _ in range(10):
        optimizer.zero_grad()
        output = model(**inputs)[0]
        loss = loss_fn(output, dummy_target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(num_iters):
        optimizer.zero_grad()
        output = model(**inputs)[0]
        loss = loss_fn(output, dummy_target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    return (end - start) / num_iters

# Inference benchmark for PyTorch XLA
def benchmark_inference_torch_xla(model, inputs, num_iters=100):
    # Warm-up
    for _ in range(10):
        _ = model(**inputs)
        xm.mark_step()  # make sure XLA device steps are executed
    start = time.time()
    for _ in range(num_iters):
        _ = model(**inputs)
        xm.mark_step()
    end = time.time()
    return (end - start) / num_iters

# Training benchmark for PyTorch XLA
def benchmark_training_torch_xla(model, inputs, num_iters=100):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        sample_output = model(**inputs)[0]
    dummy_target = torch.randn_like(sample_output)
    for _ in range(10):
        optimizer.zero_grad()
        output = model(**inputs)[0]
        loss = loss_fn(output, dummy_target)
        loss.backward()
        optimizer.step()
        xm.mark_step()
    start = time.time()
    for _ in range(num_iters):
        optimizer.zero_grad()
        output = model(**inputs)[0]
        loss = loss_fn(output, dummy_target)
        loss.backward()
        optimizer.step()
        xm.mark_step()
    end = time.time()
    return (end - start) / num_iters

# Inference benchmark for JAX/Flax (XLA compiled via jax.jit)
def benchmark_inference_flax(model, inputs, num_iters=100):
    # Define a simple forward function; Flax models use __call__
    def forward_fn(params, **batch):
        return model(**batch, params=params, train=False)[0]
    # Initialize parameters (Flax models already have params loaded)
    params = model.params
    # JIT compile the function
    jitted_forward = jax.jit(forward_fn)
    # Convert inputs to numpy arrays (JAX uses numpy arrays)
    inputs_np = {k: v.cpu().numpy() for k, v in inputs.items()}
    # Warm-up
    for _ in range(10):
        _ = jitted_forward(params, **inputs_np)
    start = time.time()
    for _ in range(num_iters):
        _ = jitted_forward(params, **inputs_np)
    end = time.time()
    return (end - start) / num_iters

# Training benchmark for JAX/Flax (simplified example)
def benchmark_training_flax(model, inputs, num_iters=100):
    # Update loss_fn to accept dropout_rng and pass it to the model.
    def loss_fn(params, batch, dropout_rng):
        # Pass dropout_rng to the model call for training mode.
        outputs = model(**batch, params=params, train=True, dropout_rng=dropout_rng)[0]
        # Dummy target with the same shape as outputs.
        dummy_target = jnp.zeros_like(outputs)
        return jnp.mean((outputs - dummy_target) ** 2)

    # JIT compile the loss function and its gradient.
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    params = model.params
    inputs_np = {k: v.cpu().numpy() for k, v in inputs.items()}
    rng = jax.random.PRNGKey(0)
    
    # Warm-up iterations.
    for _ in range(10):
        rng, dropout_rng = jax.random.split(rng)
        loss, grads = grad_fn(params, inputs_np, dropout_rng)
    
    start = time.time()
    for _ in range(num_iters):
        rng, dropout_rng = jax.random.split(rng)
        loss, grads = grad_fn(params, inputs_np, dropout_rng)
    end = time.time()
    return (end - start) / num_iters

def run_benchmarks(mapping_path):
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    all_results = {}

    # Wrap the model loop in tqdm
    for original_name, spec in tqdm(mapping.items(),
                                    desc="Models",
                                    unit="model"):
        class_name = spec["class"]
        model_id   = spec["pretrained_model"]
        all_results[original_name] = {}

        tmp_cache = tempfile.mkdtemp()
        try:
            try:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=tmp_cache)
                except ValueError:
                    # fall back to slow
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id,
                        cache_dir=tmp_cache,
                        use_fast=False
                    )
                dummy_inputs = get_dummy_inputs(tokenizer)

                ModelClass = getattr(transformers, class_name, None) or AutoModel
                pt_model   = ModelClass.from_pretrained(model_id, cache_dir=tmp_cache).eval()

                if getattr(pt_model.config, "is_encoder_decoder", False):
                    # pick the right shift function
                    if pt_model.config.model_type == "t5":
                        from transformers.models.t5.modeling_t5 import shift_tokens_right
                    else:
                        from transformers.models.bart.modeling_bart import shift_tokens_right

                    # build decoder_input_ids from the encoder dummy input_ids
                    decoder_input_ids = shift_tokens_right(
                        dummy_inputs["input_ids"],
                        pt_model.config.pad_token_id,
                        pt_model.config.decoder_start_token_id
                    )
                    dummy_inputs["decoder_input_ids"] = decoder_input_ids

                backends = ["inductor", "eager", "cudagraphs", "onnxrt", "openxla", "tvm"]
                # backends = ["onnxrt"]
                # Wrap the backend loop in tqdm, too
                for backend in tqdm(backends,
                                    desc=f"{original_name[:15]}… backends",
                                    leave=False,
                                    unit="backend"):
                    key = f"pytorch_torch_compile_{backend}"
                    try:
                        compiled = torch.compile(pt_model, backend=backend)
                        inf_t    = benchmark_inference_torch_compile(compiled, dummy_inputs)
                        train_t  = benchmark_training_torch_compile(compiled, dummy_inputs)
                        all_results[original_name][key] = {
                            "inference_s": inf_t,
                            "training_s":  train_t
                        }
                    except Exception as e:
                        all_results[original_name][key] = {"error": str(e)}

                # # JAX/Flax
                # try:
                #     flax_model = FlaxAutoModel.from_pretrained(model_id,
                #                                             cache_dir=tmp_cache)
                #     inf_f = benchmark_inference_flax(flax_model, dummy_inputs)
                #     train_f = benchmark_training_flax(flax_model, dummy_inputs)
                #     all_results[original_name]["jax_flax_xla"] = {
                #         "inference_s": inf_f,
                #         "training_s":  train_f
                #     }
                # except Exception as e:
                #     all_results[original_name]["jax_flax_xla"] = {"error": str(e)}

                append_results({original_name: all_results[original_name]})
            except Exception as e:
                # Record the failure for this model, then continue
                all_results[original_name] = {"error": repr(e)}
                append_results({original_name: all_results[original_name]})

        finally:
            shutil.rmtree(tmp_cache)

    return all_results


if __name__ == "__main__":
    json_path = "sentencepiece_models.json"  # Your JSON file containing model definitions
    bench_results = run_benchmarks(json_path)
    print("Benchmarking complete. Results:")
    print(bench_results)