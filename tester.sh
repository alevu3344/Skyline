#!/bin/bash

# Check if the datasets directory exists
if [ ! -d "datasets" ]; then
    echo "Error: datasets directory not found!"
    exit 1
fi

# Define the number of iterations
NUM_RUNS=10

# Delete the benchmark files if they exist
rm -f omp-benchmark.out
rm -f mpi-benchmark.out

# Output files
omp_output="omp-benchmark.out"
mpi_output="mpi-benchmark.out"

# Initialize output files
echo "Running OpenMP Tests" > "$omp_output"
echo "Running OpenMPI Tests" > "$mpi_output"

# Loop over each input file in the datasets directory
for input_file in datasets/*.in; do
    # Check if the file is actually a regular file
    if [ ! -f "$input_file" ]; then
        continue
    fi
    
    echo "Running tests for input: $input_file"
    
    # Initialize total time variables
    total_time_omp=0
    total_time_mpi=0

# Loop through processor counts from 1 to 8
for num_proc in $(seq 1 8); do
    echo "Running tests with $num_proc processors..."

    # Reset totals for each processor count
    total_time_omp=0
    total_time_mpi=0

    # Run the OpenMP version NUM_RUNS times
    for i in $(seq 1 $NUM_RUNS); do
        echo "  Run $i (OpenMP) with $num_proc processors..."
        # Set the number of threads for OpenMP
        export OMP_NUM_THREADS=$num_proc
        # Capture stderr output to a variable and redirect stdout to the file
        stderr_output=$(./omp-skyline < "$input_file" 2>&1 > /dev/null)

        exec_time=$(echo "$stderr_output" | grep "Execution time (s)" | awk '{print $4}')
        
        # Check if exec_time is empty and skip if no time is found
        if [ -z "$exec_time" ]; then
            echo "Error: Could not extract execution time for OpenMP (Run $i)"
            continue
        fi

        total_time_omp=$(echo "$total_time_omp + $exec_time" | bc)

        # Log the time to the OpenMP output file
        echo "Run $i (OpenMP) for $input_file with $num_proc processors: $exec_time seconds" >> "$omp_output"
    done

    # Run the OpenMPI version NUM_RUNS times
    for i in $(seq 1 $NUM_RUNS); do
        echo "  Run $i (OpenMPI) with $num_proc processors..."
        # Capture stderr output to a variable and redirect stdout to the file
        stderr_output=$(mpirun --use-hwthread-cpus -np $num_proc mpi-skyline "$input_file" 2>&1 > /dev/null)

        exec_time=$(echo "$stderr_output" | grep "Execution time (s)" | awk '{print $4}')
        
        # Check if exec_time is empty and skip if no time is found
        if [ -z "$exec_time" ]; then
            echo "Error: Could not extract execution time for OpenMPI (Run $i)"
            continue
        fi

        total_time_mpi=$(echo "$total_time_mpi + $exec_time" | bc)

        # Log the time to the OpenMPI output file
        echo "Run $i (OpenMPI) for $input_file with $num_proc processors: $exec_time seconds" >> "$mpi_output"
    done

    # Calculate averages for the current processor count
    avg_time_omp=$(echo "$total_time_omp / $NUM_RUNS" | bc -l)
    avg_time_mpi=$(echo "$total_time_mpi / $NUM_RUNS" | bc -l)

    # Format the average times
    formatted_avg_time_omp=$(printf "%0.6f" $avg_time_omp)
    formatted_avg_time_mpi=$(printf "%0.6f" $avg_time_mpi)

    # Append formatted average times to the corresponding output files
    echo "Average time for OpenMP for $input_file with $num_proc processors: $formatted_avg_time_omp seconds" >> "$omp_output"
    echo "Average time for OpenMPI for $input_file with $num_proc processors: $formatted_avg_time_mpi seconds" >> "$mpi_output"
    
    echo "------------------------------------" >> "$omp_output"
    echo "------------------------------------" >> "$mpi_output"
done

done
