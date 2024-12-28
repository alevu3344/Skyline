#!/bin/bash

# Check if the datasets directory exists
if [ ! -d "datasets" ]; then
    echo "Error: datasets directory not found!"
    exit 1
fi

# Define the number of iterations and processors
NUM_RUNS=2
num_proc=4  # Set the number of processors (for OpenMP and OpenMPI)

# Loop over each input file in the datasets directory
for input_file in datasets/*.in; do
    # Check if the file is actually a regular file
    if [ ! -f "$input_file" ]; then
        continue
    fi
    
    echo "Running tests for input: $input_file"
    
    # Initialize total time variables
    total_time_serial=0
    total_time_omp=0
    total_time_mpi=0

    # Run the serial version NUM_RUNS times
    for i in $(seq 1 $NUM_RUNS); do
        echo "  Run $i (Serial)..."
        # Capture stderr output to a variable
        stderr_output=$(./skyline < "$input_file" 2>&1 > output.out)
        
        # Debug: Echo the stderr output for inspection
        echo "Debug (Serial Run $i): $stderr_output"
        
        # Extract the execution time from the stderr output
        exec_time=$(echo "$stderr_output" | grep "Execution time (s)" | awk '{print $4}')
        
        # Check if exec_time is empty and skip if no time is found
        if [ -z "$exec_time" ]; then
            echo "Error: Could not extract execution time for Serial (Run $i)"
            continue
        fi

        total_time_serial=$(echo "$total_time_serial + $exec_time" | bc)
    done

    # Run the OpenMP version NUM_RUNS times
    for i in $(seq 1 $NUM_RUNS); do
        echo "  Run $i (OpenMP) with $num_proc processors..."
        # Set the number of threads for OpenMP
        export OMP_NUM_THREADS=$num_proc
        # Capture stderr output to a variable
        stderr_output=$(./omp-skyline < "$input_file" 2>&1 > output.out)
        
        # Debug: Echo the stderr output for inspection
        echo "Debug (OpenMP Run $i): $stderr_output"
        
        exec_time=$(echo "$stderr_output" | grep "Execution time (s)" | awk '{print $4}')
        
        # Check if exec_time is empty and skip if no time is found
        if [ -z "$exec_time" ]; then
            echo "Error: Could not extract execution time for OpenMP (Run $i)"
            continue
        fi

        total_time_omp=$(echo "$total_time_omp + $exec_time" | bc)
    done

    # Run the OpenMPI version NUM_RUNS times
    for i in $(seq 1 $NUM_RUNS); do
        echo "  Run $i (OpenMPI) with $num_proc processors..."
        # Capture stderr output to a variable
       
        stderr_output=$(mpirun --use-hwthread-cpus -np $num_proc mpi-skyline "$input_file" 2>&1 > output.out)
        
        # Debug: Echo the stderr output for inspection
        echo "Debug (OpenMPI Run $i): $stderr_output"
        
        exec_time=$(echo "$stderr_output" | grep "Execution time (s)" | awk '{print $4}')
        
        # Check if exec_time is empty and skip if no time is found
        if [ -z "$exec_time" ]; then
            echo "Error: Could not extract execution time for OpenMPI (Run $i)"
            continue
        fi

        total_time_mpi=$(echo "$total_time_mpi + $exec_time" | bc)
    done

    # Calculate averages
    avg_time_serial=$(echo "$total_time_serial / $NUM_RUNS" | bc -l)
    avg_time_omp=$(echo "$total_time_omp / $NUM_RUNS" | bc -l)
    avg_time_mpi=$(echo "$total_time_mpi / $NUM_RUNS" | bc -l)

    # Print results for this input file
    echo "Average time for serial: $avg_time_serial seconds"
    echo "Average time for OpenMP: $avg_time_omp seconds"
    echo "Average time for OpenMPI: $avg_time_mpi seconds"
    echo "------------------------------------"

done
