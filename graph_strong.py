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

    # Create a plot: average time vs. number of processors
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_processors, sorted_avg_times, marker='o', linestyle='-', color='red', label="Avg. Time")

    # Annotate each point with its execution time
    for p, t in zip(sorted_processors, sorted_avg_times):
        plt.annotate(f'{t:.2f}s', xy=(p, t), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)

    plt.xlabel("Number of Processors")
    plt.ylabel("Average Execution Time (s)")
    plt.title("Strong Scaling: Execution Time vs. Number of Processors")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Save the figure as a PDF to the /graphs directory
    output_dir = "./graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(json_file))[0] + ".pdf")
    plt.savefig(output_file, format="pdf")
    print(f"Figure saved as: {output_file}")

if __name__ == '__main__':
    main()
