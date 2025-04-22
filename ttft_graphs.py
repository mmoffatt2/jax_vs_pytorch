import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the JSON data
with open("paper_model_results_ttft.json") as f:
    data = json.load(f)

# Extract TTFT results
ttft_data = defaultdict(dict)

for entry in data:
    for model, results in entry.items():
        for backend, metrics in results.items():
            if isinstance(metrics, dict) and "ttft_s" in metrics:
                ttft_data[model][backend] = metrics["ttft_s"]

# Plot and close each figure to avoid memory warning
for model, backend_data in ttft_data.items():
    plt.figure(figsize=(10, 5))
    plt.bar(backend_data.keys(), backend_data.values())
    plt.ylabel("TTFT (s)")
    plt.title(f"TTFT for {model}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"ttft_{model}.png")  # Optional: save figure to file
    plt.close()  # <== this line clears the figure from memory
