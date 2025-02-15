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

    # Extract data: number of processors and average_time
    processors = []
    avg_times = []

    for res in results:
        processors.append(res.get("num_processors", 0))
        avg_times.append(float(res.get("average_time", 0)))

    # Sort by number of processors (just in case)
    combined = sorted(zip(processors, avg_times), key=lambda x: x[0])
    sorted_processors, sorted_avg_times = zip(*combined)

    # Determine the title based on the input file name
    if "mpi" in json_file.lower():
        title = "MPI: Strong Scaling"
    elif "omp" in json_file.lower():
        title = "OpenMP: Strong Scaling"
    else:
        title = "Strong Scaling"

    # Increase global font sizes and other style parameters
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 22,
        'axes.titlesize': 26,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 20,
    })

    # Create a plot: average time vs. number of processors
    plt.figure(figsize=(11, 6))  # Slightly wider figure to prevent cutoff
    plt.plot(sorted_processors, sorted_avg_times, marker='o', markersize=10, linewidth=3, linestyle='-', 
             color='red', label="Avg. Time")

    # Annotate each point with its execution time
    for p, t in zip(sorted_processors, sorted_avg_times):
        plt.annotate(f'{t:.2f}s', xy=(p, t), xytext=(0, 10), textcoords='offset points', 
                     ha='center', fontsize=16, weight='bold')

    plt.xlabel("Numero di processori", labelpad=15)
    plt.ylabel("Tempo di esecuzione medio (s)", labelpad=15)
    plt.title(title, pad=20)
    plt.grid(True, linestyle='--', linewidth=1.5)
    plt.legend()
    plt.tight_layout()

    # Save the figure with extra padding
    output_dir = "../relazioneTex/graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(json_file))[0] + ".pdf")
    plt.savefig(output_file, format="pdf", bbox_inches='tight')  # Ensures text isn't cut off
    print(f"Figure saved as: {output_file}")

if __name__ == '__main__':
    main()
