import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sns.set(style="whitegrid")

def load_data(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def analyze(data):
    inference_times = defaultdict(dict)
    training_times = defaultdict(dict)
    backend_failures = defaultdict(int)
    model_failures = defaultdict(int)

    for model_block in data:
        model_name, results = next(iter(model_block.items()))

        for backend, metrics in results.items():
            if not isinstance(metrics, dict) or "error" in metrics:
                backend_failures[backend] += 1
                model_failures[model_name] += 1
                continue

            inference_times[model_name][backend] = metrics.get("inference_s")
            training_times[model_name][backend] = metrics.get("training_s")

    return inference_times, training_times, backend_failures, model_failures

def plot_bar_chart(data_dict, title, ylabel, filename):
    plt.figure(figsize=(12, 6))
    for model_name, timings in data_dict.items():
        x = list(timings.keys())
        y = list(timings.values())
        plt.plot(x, y, marker='o', label=model_name)

    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def compute_average_times(time_dict):
    backend_totals = defaultdict(float)
    backend_counts = defaultdict(int)

    for model_timings in time_dict.values():
        for backend, time in model_timings.items():
            if time is not None:
                backend_totals[backend] += time
                backend_counts[backend] += 1

    backend_averages = {
        backend: backend_totals[backend] / backend_counts[backend]
        for backend in backend_totals
    }
    return backend_averages

def plot_average_times(avg_dict, title, ylabel, filename):
    items = sorted(avg_dict.items(), key=lambda x: x[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=labels, y=values, palette="Blues_d")
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)

    # Add value annotations on bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_failure_counts(count_dict, title, filename):
    items = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    counts = [v for _, v in items]

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=labels, y=counts, palette="Reds_d")
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Failure Count", fontsize=12)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)

    for i, v in enumerate(counts):
        ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    data = load_data("paper_model_results_updated4.json")  # Or your preferred file
    inference_times, training_times, backend_failures, model_failures = analyze(data)

    plot_bar_chart(inference_times, "Inference Time per Backend", "Seconds", "inference_times.png")
    plot_bar_chart(training_times, "Training Time per Backend", "Seconds", "training_times.png")
    
    avg_inference = compute_average_times(inference_times)
    avg_training = compute_average_times(training_times)
    plot_average_times(avg_inference, "Average Inference Time per Backend", "Seconds", "avg_inference_times.png")
    plot_average_times(avg_training, "Average Training Time per Backend", "Seconds", "avg_training_times.png")
    
    plot_failure_counts(backend_failures, "Backend Failure Counts", "backend_failures.png")
    plot_failure_counts(model_failures, "Model Failure Counts", "model_failures.png")

    print("✅ Plots saved as:")
    print(" - inference_times.png")
    print(" - training_times.png")
    print(" - avg_inference_times.png")
    print(" - avg_training_times.png")
    print(" - backend_failures.png")
    print(" - model_failures.png")

if __name__ == "__main__":
    main()
