#!/bin/bash

# Check if the datasets directory exists
if [ ! -d "datasets" ]; then
    echo "Error: datasets directory not found!"
    exit 1
fi

# Define the number of iterations
NUM_RUNS=1

# Delete the benchmark file if it exists
omp_output="omp_strong_scaling.out"
rm -f "$omp_output"

# Initialize the output file
echo "Running OpenMP Strong Scaling Tests" > "$omp_output"

# Fixed input file
input_file="datasets/strong_scaling.in"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file $input_file not found!"
    exit 1
fi

# Loop through processor counts
for num_proc in $(seq 1 8); do
    echo "Running OpenMP tests with $num_proc processors..."

    # Initialize total time variable for the current processor count
    total_time_omp=0

    # Run the OpenMP version NUM_RUNS times
    for i in $(seq 1 $NUM_RUNS); do
        echo "  Run $i (OpenMP) with $num_proc processors..."
        # Set the number of threads for OpenMP
        export OMP_NUM_THREADS=$num_proc
        # Capture stderr output to a variable and redirect stdout to /dev/null
        stderr_output=$(./omp-skyline < "$input_file" 2>&1 > /dev/null)

        exec_time=$(echo "$stderr_output" | grep "Execution time (s)" | awk '{print $4}')
        
        # Check if exec_time is empty and skip if no time is found
        if [ -z "$exec_time" ]; then
            echo "Error: Could not extract execution time for OpenMP (Run $i)"
            continue
        fi

        total_time_omp=$(echo "$total_time_omp + $exec_time" | bc)

        # Log the time to the OpenMP output file
        echo "Run $i (OpenMP) with $num_proc processors: $exec_time seconds" >> "$omp_output"
    done

    # Calculate the average for the current processor count
    avg_time_omp=$(echo "$total_time_omp / $NUM_RUNS" | bc -l)

    # Format the average time
    formatted_avg_time_omp=$(printf "%0.6f" $avg_time_omp)

    # Append formatted average time to the output file
    echo "Average time for OpenMP with $num_proc processors: $formatted_avg_time_omp seconds" >> "$omp_output"
    echo "------------------------------------" >> "$omp_output"
done
