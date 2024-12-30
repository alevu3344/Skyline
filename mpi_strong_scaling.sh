#!/bin/bash

# Check if the datasets directory exists
if [ ! -d "datasets" ]; then
    echo "Error: datasets directory not found!"
    exit 1
fi

# Define the number of iterations
NUM_RUNS=1

# Delete the benchmark file if it exists
mpi_output="mpi_strong_scaling.out"
rm -f "$mpi_output"

# Initialize the output file
echo "Running MPI Strong Scaling Tests" > "$mpi_output"

# Fixed input file
input_file="datasets/strong_scaling.in"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file $input_file not found!"
    exit 1
fi

# Loop through processor counts
for num_proc in $(seq 1 8); do
    echo "Running MPI tests with $num_proc processors..."

    # Initialize total time variable for the current processor count
    total_time_mpi=0

    # Run the MPI version NUM_RUNS times
    for i in $(seq 1 $NUM_RUNS); do
        echo "  Run $i (MPI) with $num_proc processors..."
        # Capture stderr output to a variable and redirect stdout to /dev/null
        stderr_output=$(mpirun --use-hwthread-cpus -np $num_proc mpi-skyline "$input_file" 2>&1 > /dev/null)

        exec_time=$(echo "$stderr_output" | grep "Execution time (s)" | awk '{print $4}')
        
        # Check if exec_time is empty and skip if no time is found
        if [ -z "$exec_time" ]; then
            echo "Error: Could not extract execution time for MPI (Run $i)"
            continue
        fi

        total_time_mpi=$(echo "$total_time_mpi + $exec_time" | bc)

        # Log the time to the MPI output file
        echo "Run $i (MPI) with $num_proc processors: $exec_time seconds" >> "$mpi_output"
    done

    # Calculate the average for the current processor count
    avg_time_mpi=$(echo "$total_time_mpi / $NUM_RUNS" | bc -l)

    # Format the average time
    formatted_avg_time_mpi=$(printf "%0.6f" $avg_time_mpi)

    # Append formatted average time to the output file
    echo "Average time for MPI with $num_proc processors: $formatted_avg_time_mpi seconds" >> "$mpi_output"
    echo "------------------------------------" >> "$mpi_output"
done
