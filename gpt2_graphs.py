import json
import matplotlib.pyplot as plt
import numpy as np

# Load JSON data
with open("gpt2_ablation.json") as f:
    data = json.load(f)

models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
color_map = dict(zip(models, colors))

plt.figure(figsize=(12, 8))

for model in models:
    model_data = data.get(model, {})
    color = color_map[model]

    # Get and trim eager data
    eager_data = model_data.get("eager", [])
    eager_data = [d for d in eager_data if d["iterations"] <= 2000]
    x_eager = [d["iterations"] for d in eager_data]
    y_eager = [d["inference_time_s"] for d in eager_data]

    # Get and trim compiled data
    compiled_key = next((k for k in model_data if k.startswith("benchmark_iterations_")), None)
    if not compiled_key:
        continue
    compiled_data = model_data[compiled_key]
    compiled_data = [d for d in compiled_data if d["iterations"] <= 2000]
    x_compiled = [d["iterations"] for d in compiled_data]
    y_compiled = [d["total_time_s"] for d in compiled_data]

    # Plot both with the same color but different styles
    plt.plot(x_eager, y_eager, linestyle='--', color=color, label=f"{model} (eager)")
    plt.plot(x_compiled, y_compiled, linestyle='-',  color=color, label=f"{model} (compiled)")

    # Find closest intersection
    differences = np.array(y_compiled) - np.array(y_eager)
    closest_idx = np.argmin(np.abs(differences))
    ix = x_compiled[closest_idx]
    iy = y_compiled[closest_idx]

    # Mark and annotate intersection
    plt.plot(ix, iy, 'o', color=color)
    plt.text(ix, iy + 11, f"{ix}", fontsize=15, ha='center', color=color)

# Labels and save
plt.xlabel("Iterations", fontsize=20)
plt.ylabel("Total Time (s)", fontsize=20)
plt.title("Eager vs Compiled Inference Time", fontsize=24)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.savefig("gpt2_eager_vs_compiled_colored.png")
