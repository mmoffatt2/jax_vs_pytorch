import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Path to your JSON file
INPUT_FILE = "paper_model_results_updated5.json"

# Load data
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

ttft_data = defaultdict(dict)
all_backends = set()

# Parse JSON
for entry in data:
    for model, results in entry.items():
        for backend, metrics in results.items():
            if isinstance(metrics, dict) and "inference_s" in metrics and "training_s" in metrics:
                ttft_data[model][backend] = metrics
                all_backends.add(backend)

backend_order = sorted(all_backends)

# Filter models with complete backend coverage
filtered_models = [m for m in ttft_data if all(b in ttft_data[m] for b in backend_order)]

# Compute average times for filtered models
inference_data = defaultdict(list)
training_data = defaultdict(list)

for model in filtered_models:
    for backend in backend_order:
        inference_data[backend].append(ttft_data[model][backend]["inference_s"])
        training_data[backend].append(ttft_data[model][backend]["training_s"])

avg_inf = {b: np.mean(inference_data[b]) for b in backend_order}
avg_train = {b: np.mean(training_data[b]) for b in backend_order}

# Plotting function
def plot_bar(data, title, ylabel, filename):
    x = np.arange(len(backend_order))
    width = 0.5

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, [data[b] for b in backend_order], width)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(backend_order, rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

# Generate plots
plot_bar(avg_inf, "Average Inference Time (Filtered Models)", "Inference Time (s)", "inference_filtered.png")
plot_bar(avg_train, "Average Training Time (Filtered Models)", "Training Time (s)", "training_filtered.png")
