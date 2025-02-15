#!/bin/bash

# Check if the datasets directory exists
if [ ! -d "datasets" ]; then
    echo "Error: datasets directory not found!"
    exit 1
fi

# Define the number of iterations
NUM_RUNS=20

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

# Compute the number of points using weak scaling formula: N' = BASE_N * sqrt(num_proc)
num_points=$BASE_N

# Generate the input file
input_file="datasets/strong_scaling.in"
echo "Generating dataset with $num_points points for strong scaling..."
$INPUTGEN "$num_points" "$DIMENSIONS" >"$input_file"

# Check if the input file was generated
if [ ! -f "$input_file" ]; then
    echo "Error: no $input_file for $num_proc processors!" >&2
    continue
fi

# Define the JSON output file and remove it if it exists
json_output="omp_strong_scaling.json"
rm -f "$json_output"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file $input_file not found!"
    exit 1
fi

# Initialize the JSON structure
echo "{" >"$json_output"
echo "  \"results\": [" >>"$json_output"

first_entry=1

# Loop through processor counts
for num_proc in $(seq 1 8); do
    echo "Running OpenMP tests with $num_proc processors..."

    total_time_omp=0
    run_results=""

    first_run=1

    # Run the OpenMP version NUM_RUNS times
    for i in $(seq 1 $NUM_RUNS); do

        echo "  Run $i (OpenMP) with $num_proc processors..."
        export OMP_NUM_THREADS=$num_proc
        # Capture stderr output to a variable and redirect stdout to /dev/null
        stderr_output=$(./omp-skyline <"$input_file" 2>&1 >/dev/null)

        exec_time=$(echo "$stderr_output" | grep "Execution time (s)" | awk '{print $4}')

        # Check if exec_time is empty and skip if no time is found
        if [ -z "$exec_time" ]; then
            echo "Error: Could not extract execution time for OpenMP (Run $i)"
            continue
        fi

        total_time_omp=$(echo "$total_time_omp + $exec_time" | bc)

        # Build JSON snippet for the current run
        run_json="{\"run\": $i, \"time\": $exec_time}"
        if [ $first_run -eq 1 ]; then
            run_results="$run_json"
            first_run=0
        else
            run_results+=", $run_json"
        fi
    done

    # Calculate the average for the current processor count
    avg_time_omp=$(echo "$total_time_omp / $NUM_RUNS" | bc -l)
    formatted_avg_time_omp=$(printf "%0.6f" $avg_time_omp)

    # Build JSON entry for the current processor count
    json_entry="    {\"num_processors\": $num_proc, \"runs\": [$run_results], \"average_time\": $formatted_avg_time_omp}"

    # Append comma for all but the first entry
    if [ $first_entry -eq 1 ]; then
        echo "$json_entry" >>"$json_output"
        first_entry=0
    else
        echo "    ,$json_entry" >>"$json_output"
    fi
done

# Close the JSON array and the JSON object
echo "  ]" >>"$json_output"
echo "}" >>"$json_output"

echo "JSON output saved to $json_output"

source ~/anaconda3/etc/profile.d/conda.sh # Adjust this path to match your Conda installation

conda activate

# Generate the graphs by executing the Python script
python -u graph_strong.py "$json_output"

conda deactivate
