import json
import time
import torch
from transformers import AutoModel, AutoTokenizer, FlaxAutoModel
import torch_xla.core.xla_model as xm  # for PyTorch XLA
import jax
import jax.numpy as jnp

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
    # For training, we define a loss function and compute gradients.
    def loss_fn(params, batch):
        outputs = model(**batch, params=params, train=True)[0]
        # Dummy target: same shape as outputs
        dummy_target = jnp.zeros_like(outputs)
        return jnp.mean((outputs - dummy_target) ** 2)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    params = model.params
    inputs_np = {k: v.cpu().numpy() for k, v in inputs.items()}
    # Warm-up
    for _ in range(10):
        loss, grads = grad_fn(params, inputs_np)
    start = time.time()
    for _ in range(num_iters):
        loss, grads = grad_fn(params, inputs_np)
    end = time.time()
    return (end - start) / num_iters

# Main function that reads a JSON file of models and benchmarks them
def run_benchmarks(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    
    results = {}
    
    for model_entry in config["models"]:
        model_name = model_entry["name"]
        print(f"Benchmarking {model_name}...")
        results[model_name] = {}

        # Create tokenizer and dummy input for PyTorch (will also be reused for Flax)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dummy_inputs = get_dummy_inputs(tokenizer)
        
        # --- PyTorch with torch.compile ---
        model_pt = AutoModel.from_pretrained(model_name)
        model_pt.eval()
        compiled_model = torch.compile(model_pt)
        inf_time = benchmark_inference_torch_compile(compiled_model, dummy_inputs)
        train_time = benchmark_training_torch_compile(compiled_model, dummy_inputs)
        results[model_name]["pytorch_torch_compile"] = {"inference_time": inf_time,
                                                        "training_time": train_time}
        
        # --- PyTorch with XLA ---
        # Note: Ensure you have an XLA-supported device (e.g. TPU or a GPU configured with torch_xla)
        device = xm.xla_device()
        model_xla = AutoModel.from_pretrained(model_name).to(device)
        model_xla.eval()
        dummy_inputs_xla = get_dummy_inputs(tokenizer, device=device)
        inf_time_xla = benchmark_inference_torch_xla(model_xla, dummy_inputs_xla)
        train_time_xla = benchmark_training_torch_xla(model_xla, dummy_inputs_xla)
        results[model_name]["pytorch_xla"] = {"inference_time": inf_time_xla,
                                              "training_time": train_time_xla}
        
        # --- JAX/Flax with XLA ---
        model_flax = FlaxAutoModel.from_pretrained(model_name)
        inf_time_flax = benchmark_inference_flax(model_flax, dummy_inputs)
        train_time_flax = benchmark_training_flax(model_flax, dummy_inputs)
        results[model_name]["jax_flax_xla"] = {"inference_time": inf_time_flax,
                                               "training_time": train_time_flax}
        print(f"Results for {model_name}: {results[model_name]}")
    
    return results

if __name__ == "__main__":
    json_path = "models.json"  # Your JSON file containing model definitions
    bench_results = run_benchmarks(json_path)
    print("Benchmarking complete. Results:")
    print(bench_results)
