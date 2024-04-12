#!/bin/bash

# Get the job ID of the last completed job
last_job_id=$(sacct --format=JobID -n -u $USER -S "$(date -d yesterday +"%Y-%m-%d")" -o JobID | tail -n 1)

# Check if a job ID was found
if [ -z "$last_job_id" ]; then
    echo "No completed job found."
    exit 1
fi

# Get the job information
job_info=$(scontrol show job $last_job_id)

# Determine the filename
filename="${1:-job_info.txt}"

# Save the job information to the specified file
echo "$job_info" > "$filename"

echo "Job information saved to $filename."
