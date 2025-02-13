#!/bin/bash

# Check if the datasets directory exists
if [ ! -d "datasets" ]; then
    echo "Error: datasets directory not found!"
    exit 1
fi

# Define the number of iterations
NUM_RUNS=5

# Delete the previous MPI output file if it exists
mpi_output="mpi_weak_scaling.json"
rm -f "$mpi_output"

# Begin JSON output with a top-level "results" key
echo "{" > "$mpi_output"
echo '  "results": [' >> "$mpi_output"

first_result=true

# Loop through processor counts and corresponding input files
for num_proc in $(seq 1 8); do
    input_file="datasets/weak_scaling_${num_proc}_proc.in"
    
    # Check if the input file exists
    if [ ! -f "$input_file" ]; then
        echo "Error: Input file $input_file not found for $num_proc processors!" >&2
        continue
    fi

    # Extract number of points from the input file (assuming it's on the second line)
    num_points=$(sed -n '2p' "$input_file" | tr -d '\r\n')

    total_time_mpi=0
    runs_json=""   # Will accumulate JSON objects for each run
    first_run=true

    # Run the OpenMPI version NUM_RUNS times
    for i in $(seq 1 $NUM_RUNS); do
        echo "  Run $i (OpenMPI) with $num_proc processors..."
        # Execute the mpi-skyline executable using mpirun and capture stderr output
        stderr_output=$(mpirun --use-hwthread-cpus -np $num_proc mpi-skyline "$input_file" 2>&1 >/dev/null)

        exec_time=$(echo "$stderr_output" | grep "Execution time (s)" | awk '{print $4}')
        
        # Check if exec_time is empty and skip if no time is found
        if [ -z "$exec_time" ]; then
            echo "Error: Could not extract execution time for OpenMPI (Run $i)" >&2
            continue
        fi

        total_time_mpi=$(echo "$total_time_mpi + $exec_time" | bc)

        # Build the JSON object for this run
        run_json="{\"run\": $i, \"time\": $exec_time}"
        if [ "$first_run" = true ]; then
            runs_json="$run_json"
            first_run=false
        else
            runs_json="$runs_json, $run_json"
        fi
    done

    # Calculate the average execution time for this processor count
    avg_time_mpi=$(echo "$total_time_mpi / $NUM_RUNS" | bc -l)
    formatted_avg_time_mpi=$(printf "%.6f" "$avg_time_mpi")

    # Build the JSON object for this result
    json_obj=$(cat <<EOF
    {
      "num_processors": $num_proc,
      "num_points": $num_points,
      "runs": [ $runs_json ],
      "average_time": $formatted_avg_time_mpi
    }
EOF
)

    # Append a comma if this is not the first result for valid JSON syntax
    if [ "$first_result" = true ]; then
        echo "$json_obj" >> "$mpi_output"
        first_result=false
    else
        echo ",$json_obj" >> "$mpi_output"
    fi
done

# End the JSON array and top-level object
echo "  ]" >> "$mpi_output"
echo "}" >> "$mpi_output"
