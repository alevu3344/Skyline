#!/usr/bin/env python3
import json
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: {} <json_file>".format(sys.argv[0]))
        sys.exit(1)

    input_file = sys.argv[1]

    # Load JSON data from file
    with open(input_file, 'r') as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("No results found in the JSON file.")
        sys.exit(1)

    # Extract average times for each processor count
    proc_times = {}
    for res in results:
        p = res.get("num_processors", 0)
        avg_time = float(res.get("average_time", 0))
        proc_times[p] = avg_time

    # Ensure the result for 1 processor is present
    if 1 not in proc_times:
        print("No result for 1 processor found. Cannot compute speedup.")
        sys.exit(1)

    T1 = proc_times[1]
    sorted_procs = sorted(proc_times.keys())

    # Prepare table rows in LaTeX format
    lines = []
    for p in sorted_procs:
        T_p = proc_times[p]
        if T_p == 0:
            speedup = float('inf')
            efficiency = 0
        else:
            speedup = T1 / T_p
            efficiency = speedup / p

        # Format the values to 6 decimal places for T(p) and 3 for speedup/efficiency
        line = f"{p} & {T_p:.6f} & {speedup:.3f} & {efficiency:.3f} \\\\"
        lines.append(line)

    # Determine output filename based on input filename
    lower_name = os.path.basename(input_file).lower()
    if "mpi" in lower_name:
        out_filename = "mpi_table.txt"
    elif "omp" in lower_name:
        out_filename = "omp_table.txt"
    else:
        out_filename = "table.txt"

    # Write the LaTeX table rows to the output file
    with open(out_filename, "w") as out_file:
        for line in lines:
            out_file.write(line + "\n")

    print(f"Table data saved to: {out_filename}")

if __name__ == '__main__':
    main()
