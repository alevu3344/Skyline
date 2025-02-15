#!/usr/bin/env python3
import json
import sys
import matplotlib.pyplot as plt
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: {} <json_file>".format(sys.argv[0]))
        sys.exit(1)

    json_file = sys.argv[1]

    # Load JSON data from file
    with open(json_file, 'r') as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("No results found in the JSON file.")
        sys.exit(1)

    # Extract data: input size (num_points), average_time, and number of processors
    input_sizes = []
    avg_times = []
    procs = []
    for res in results:
        input_sizes.append(res.get("num_points", 0))
        avg_times.append(float(res.get("average_time", 0)))
        procs.append(res.get("num_processors", 0))

    # Increase global font sizes and other style parameters
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 22,
        'axes.titlesize': 26,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20,
    })

    # Determine the title based on the input file name
    if "mpi" in json_file.lower():
        title = "MPI: Weak Scaling"
    elif "omp" in json_file.lower():
        title = "OpenMP: Weak Scaling"
    else:
        title = "Weak Scaling"

    # Create a plot: Average Execution Time vs. Input Size
    plt.figure(figsize=(11, 7))
    plt.plot(input_sizes, avg_times, marker='o', markersize=12, linewidth=3,
             linestyle='-', color='blue', label="Avg. Time")

    # Annotate each point with the number of processors
    for x, y, p in zip(input_sizes, avg_times, procs):
        plt.annotate(f'{p} proc', xy=(x, y), xytext=(0, 12),
                     textcoords="offset points", ha='center', fontsize=18, fontweight='bold')

    plt.xlabel("Numero di punti", labelpad=15)
    plt.ylabel("Tempo di esecuzione medio (s)", labelpad=15)
    plt.title(title, pad=20)
    plt.grid(True, linestyle='--', linewidth=1.5)
    plt.legend()
    plt.tight_layout()

    # Save the figure as a PDF to the ./graphs directory, ensuring nothing is cut off.
    output_dir = "../relazioneTex/graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(json_file))[0] + ".pdf")
    plt.savefig(output_file, format="pdf", bbox_inches='tight')
    print(f"Figure saved as: {output_file}")

if __name__ == '__main__':
    main()
