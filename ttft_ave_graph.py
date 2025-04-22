import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Load and parse TTFT data
with open("paper_model_results_ttft.json") as f:
    data = json.load(f)

ttft_data = defaultdict(dict)
excluded_backends = {"pytorch_torch_compile_tvm", "pytorch_torch_compile_openxla"}

# Collect TTFT data and backend set
all_backends = set()
for entry in data:
    for model, results in entry.items():
        for backend, metrics in results.items():
            if isinstance(metrics, dict) and "ttft_s" in metrics:
                if backend not in excluded_backends:
                    ttft_data[model][backend] = metrics["ttft_s"]
                    all_backends.add(backend)

# Ensure consistent backend order for plotting
backend_order = sorted(all_backends)

# --- 1. Average TTFT per backend (All models, excluding the two backends) ---
average_all = defaultdict(list)
for model, backend_data in ttft_data.items():
    for backend in backend_order:
        if backend in backend_data:
            average_all[backend].append(backend_data[backend])

avg_all_means = {b: np.mean(average_all[b]) for b in backend_order}

# --- 2. Average TTFT per backend (only models that have full coverage of the selected backends) ---
valid_models = [model for model, data in ttft_data.items() if all(b in data for b in backend_order)]

average_filtered = defaultdict(list)
for model in valid_models:
    for backend in backend_order:
        average_filtered[backend].append(ttft_data[model][backend])

avg_filtered_means = {b: np.mean(average_filtered[b]) for b in backend_order}

# --- Plotting ---
def plot_avg(data_dict, title, filename):
    plt.figure(figsize=(10, 5))
    bars = plt.bar(backend_order, [data_dict[b] for b in backend_order])
    plt.ylabel("Average TTFT (s)")
    plt.title(title)
    plt.xticks(rotation=45)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f"{yval:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_avg(avg_all_means, "Average TTFT per Backend", "avg_ttft_all.png")
plot_avg(avg_filtered_means, "Average TTFT per Backend (Filtered Models, Excl. TVM & OpenXLA)", "avg_ttft_filtered.png")
