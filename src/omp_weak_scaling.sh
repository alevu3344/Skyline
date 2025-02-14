#!/bin/bash

# Delete the previous output file if it exists
omp_output="omp_weak_scaling.json"
rm -f "$omp_output"

# Begin JSON output with a top-level "results" object
echo "{" >"$omp_output"
echo '  "results": [' >>"$omp_output"

first_result=true

NUM_RUNS=10


# Base number of points for weak scaling
BASE_N=50000
DIMENSIONS=20 # Number of dimensions

# Path to the input generator executable
INPUTGEN="./datasets/inputgen"

# Check if the input generator exists
if [ ! -f "$INPUTGEN" ]; then
    echo "Error: Input generator executable '$INPUTGEN' not found!"
    exit 1
fi

# Loop through processor counts and generate inputs dynamically
for num_proc in $(seq 1 8); do

    # Compute the number of points using weak scaling formula: N' = BASE_N * sqrt(num_proc)
    num_points=$(echo "$BASE_N * sqrt($num_proc)" | bc -l)
    num_points=$(printf "%.0f" "$num_points") # Round to nearest integer

    # Generate the input file
    input_file="datasets/weak_scaling_${num_proc}_proc.in"
    echo "Generating dataset with $num_points points for $num_proc processors..."
    $INPUTGEN "$num_points" "$DIMENSIONS" >"$input_file"

    # Check if the input file was generated
    if [ ! -f "$input_file" ]; then
        echo "Error: no $input_file for $num_proc processors!" >&2
        continue
    fi

    echo "Running OpenMP tests for $input_file with $num_proc processors..."

    total_time_omp=0
    runs_json="" # This will hold the JSON for individual runs
    first_run=true

    # Run the OpenMP version NUM_RUNS times
    for i in $(seq 1 $NUM_RUNS); do
        echo "  Run $i (OpenMP) with $num_proc processors..."
        export OMP_NUM_THREADS=$num_proc
        # Capture stderr output (execution time) and redirect stdout to /dev/null
        stderr_output=$(./omp-skyline <"$input_file" 2>&1 >/dev/null)

        exec_time=$(echo "$stderr_output" | grep "Execution time (s)" | awk '{print $4}')

        # Check if exec_time is empty and skip if no time is found
        if [ -z "$exec_time" ]; then
            echo "Error: Could not extract execution time for OpenMP (Run $i)" >&2
            continue
        fi

        total_time_omp=$(echo "$total_time_omp + $exec_time" | bc)

        # Build the JSON object for this run
        run_json="{\"run\": $i, \"time\": $exec_time}"

        if [ "$first_run" = true ]; then
            runs_json="$run_json"
            first_run=false
        else
            runs_json="$runs_json, $run_json"
        fi
    done

    # Calculate the average time
    avg_time_omp=$(echo "$total_time_omp / $NUM_RUNS" | bc -l)
    formatted_avg_time_omp=$(printf "%.6f" "$avg_time_omp")

    # Build JSON object for this processor count (including num_points)
    json_obj=$(
        cat <<EOF
    {
      "num_processors": $num_proc,
      "num_points": $num_points,
      "runs": [ $runs_json ],
      "average_time": $formatted_avg_time_omp
    }
EOF
    )

    # If not the first result, add a comma to separate JSON objects
    if [ "$first_result" = true ]; then
        echo "$json_obj" >>"$omp_output"
        first_result=false
    else
        echo ",$json_obj" >>"$omp_output"
    fi
done

# End the JSON array and top-level object
echo "  ]" >>"$omp_output"
echo "}" >>"$omp_output"

source ~/anaconda3/etc/profile.d/conda.sh # Adjust this path to match your Conda installation

conda activate
# Generate the graphs by executing the Python script
python -u graph_weak.py "$omp_output"

conda deactivate
