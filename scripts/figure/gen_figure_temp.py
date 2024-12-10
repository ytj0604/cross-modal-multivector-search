import matplotlib.pyplot as plt
import pandas as pd
import os

# Function to read data from a file and draw the chart
def plot_from_file(file_path):
    # Extract details from the filename
    base_name = os.path.basename(file_path)
    parts = base_name.split('_')
    m_value = parts[0]  # Extract m value (e.g., '2')
    mode = parts[1].split('.')[0]  # Extract mode (e.g., 'i2t')

    # Read the file into a pandas DataFrame
    df = pd.read_csv(file_path, delimiter='\t')

    # Extract labels and data
    x_labels = df["Beam Width"]
    original_latency = df["Original_algorithm - Workload Latency"]
    shared_latency = df["Shared_visited_list - Workload Latency"]
    original_computation = df["Original_algorithm - #Distance Computation"]
    shared_computation = df["Shared_visited_list - #Distance Computation"]

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x_positions = range(len(x_labels))
    bar_width = 0.4

    # Bar chart for latencies
    ax1.bar(
        [x - bar_width / 2 for x in x_positions],
        original_latency,
        bar_width,
        label="Original_algorithm - Workload Latency",
        color="blue",
        alpha=0.7
    )
    ax1.bar(
        [x + bar_width / 2 for x in x_positions],
        shared_latency,
        bar_width,
        label="Shared_visited_list - Workload Latency",
        color="orange",
        alpha=0.7
    )
    ax1.set_ylabel("Workload Latency (s)")
    ax1.set_xlabel("Beam Width")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.legend(loc="upper left")

    # Line chart for distance computations with a second y-axis
    ax2 = ax1.twinx()
    ax2.plot(
        x_positions,
        original_computation,
        label="Original_algorithm - #Distance Computation",
        color="green",
        marker="o",
        linestyle="--"
    )
    ax2.plot(
        x_positions,
        shared_computation,
        label="Shared_visited_list - #Distance Computation",
        color="red",
        marker="o",
        linestyle="--"
    )
    ax2.set_ylabel("#Distance Computation")
    ax2.legend(loc="upper right")

    # Set the dynamic title
    plt.title(f"Comparison of Workload Latency and #Distance Computation for {mode} and m={m_value}")
    plt.tight_layout()

    # Save the figure in the same directory as the input file
    output_file = file_path.replace('.tsv', '.png')
    plt.savefig(output_file)
    print(f"Figure saved as: {output_file}")

    plt.show()

# Example usage with a file path
for n in [2, 3, 4, 5]:
    for st in ["t2i", "i2t"]:
        file_path = f"/mnt/tjyoon/shared_visited_list_eval/{n}_{st}.tsv"
        plot_from_file(file_path)
