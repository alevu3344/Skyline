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

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(input_sizes, avg_times, marker='o', linestyle='-', color='blue', label="Avg. Time")

    # Annotate each point with the number of processors
    for x, y, p in zip(input_sizes, avg_times, procs):
        plt.annotate(f'{p} proc', xy=(x, y), xytext=(0, 10), textcoords="offset points", ha='center', fontsize=9)

    plt.xlabel("Input Size (number of points)")
    plt.ylabel("Average Execution Time (s)")
    plt.title("Weak Scaling Performance: Execution Time vs. Input Size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the figure as a PDF to the /graphs directory
    output_dir = "./graphs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #the name of the figure will be the same as the name of the json file but with a .pdf extension
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(json_file))[0] + ".pdf")
    plt.savefig(output_file, format="pdf")
    print(f"Figure saved as: {output_file}")

if __name__ == '__main__':
    main()
