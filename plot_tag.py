import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Directory containing your gathered runs
ROOT_DIR = "tag-dcg"
# The scalar tag you want to plot
METRIC_NAME = "test_return_mean"
# Exponential smoothing factor (0 = no smoothing, 1 = max smoothing)
SMOOTHING = 0.85
# Maximum number of timesteps to display (1 million)
MAX_STEPS = 1_000_000


def ema_smooth(data, weight):
    """
    Apply exponential moving average smoothing to a 1D array.
    """
    smoothed = []
    last = data[0]
    for point in data:
        last = last * weight + (1 - weight) * point
        smoothed.append(last)
    return np.array(smoothed)


def extract_events(event_file):
    """
    Read a single TensorBoard event file and extract (step, value) arrays for METRIC_NAME,
    but only up to MAX_STEPS.
    """
    ea = EventAccumulator(event_file)
    ea.Reload()
    tags = ea.Tags()
    if 'scalars' not in tags or METRIC_NAME not in tags['scalars']:
        print(f"[warn] Metric {METRIC_NAME} not in {event_file}")
        return None

    events = ea.Scalars(METRIC_NAME)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])

    # Cut off after MAX_STEPS
    mask = steps <= MAX_STEPS
    steps = steps[mask]
    values = values[mask]

    return steps, values


def gather_experiments(root_dir):
    """
    Walk ROOT_DIR, pick up subdirectories matching your seeds,
    and collect their (steps, values) tuples grouped by algo and variant.
    """
    exp_data = defaultdict(list)
    # Pattern: gather__<algo>__rnn__seed_<n>[_<variant>]
    pattern = re.compile(r'^gymma__(?P<algo>[^_]+)__rnn__seed_(?P<seed>\d+)(?:_(?P<variant>.+))?$')

    for subdir in sorted(os.listdir(root_dir)):
        dir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(dir_path):
            continue
        m = pattern.match(subdir)
        if not m:
            continue
        base_algo = m.group('algo')
        variant = m.group('variant')
        algo_key = f"{base_algo}_{variant}" if variant else base_algo

        # Find the TensorBoard file(s)
        event_files = glob.glob(os.path.join(dir_path, 'events.out.tfevents*'))
        if not event_files:
            print(f"[warn] No event files in {dir_path}")
            continue
        # Pick the most recent file (highest timestamp)
        event_file = sorted(event_files)[-1]
        result = extract_events(event_file)
        if result is None:
            continue
        steps, values = result
        exp_data[algo_key].append((steps, values))

    return exp_data


def align_and_aggregate(runs):
    """
    Given multiple (steps, values) runs, truncate to shortest,
    compute mean+std, and smooth them.
    """
    min_len = min(len(vals) for _, vals in runs)
    steps = runs[0][0][:min_len]
    all_vals = np.stack([vals[:min_len] for _, vals in runs], axis=0)

    mean = all_vals.mean(axis=0)
    std = all_vals.std(axis=0)

    mean = ema_smooth(mean, SMOOTHING)
    std = ema_smooth(std, SMOOTHING)

    return steps, mean, std


def plot_all(exp_data):
    """
    Plot mean (with ±std) for each algorithm/variant.
    """
    # Print summary of how many runs were found per key
    print("Summary of runs per variant:")
    for algo_key, runs in sorted(exp_data.items()):
        print(f"  {algo_key}: {len(runs)} run(s)")

    plt.figure(figsize=(12, 7))
    for algo_key, runs in sorted(exp_data.items()):
        n = len(runs)
        if n == 0:
            continue
        if n == 1:
            # Plot single-run case
            print(f"[info] Only one run for {algo_key}, plotting raw curve")
            steps, values = runs[0]
            mean = ema_smooth(values, SMOOTHING)
            std = np.zeros_like(mean)
        else:
            steps, mean, std = align_and_aggregate(runs)

        plt.plot(steps, mean, label=algo_key)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

    plt.xlim(0, MAX_STEPS)
    plt.xlabel("Timesteps")
    plt.ylabel(METRIC_NAME)
    plt.title(f"Smoothed Performance (cut at {MAX_STEPS} steps): {METRIC_NAME}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("rl_results_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    data = gather_experiments(ROOT_DIR)
    plot_all(data)
