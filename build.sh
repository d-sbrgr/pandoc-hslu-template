#!/bin/bash

# Prevent Git Bash from converting Unix paths to Windows paths
export MSYS_NO_PATHCONV=1

# Docker image name
IMAGE_NAME="pandoc-hslu-report"

# Check if Docker image exists
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Docker image '$IMAGE_NAME' not found. Building it now..."
    docker build -t $IMAGE_NAME .
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build Docker image."
        exit 1
    fi
    echo "Docker image built successfully."
else
    echo "Docker image '$IMAGE_NAME' already exists."
fi

# Create _build directory if it doesn't exist
mkdir -p _build

# Convert EPS files to PDF if they exist
echo "Converting EPS files to PDF..."
for eps_file in figs/*.eps; do
    if [ -f "$eps_file" ]; then
        pdf_file="${eps_file%.eps}.pdf"
        if [ ! -f "$pdf_file" ] || [ "$eps_file" -nt "$pdf_file" ]; then
            echo "Converting $eps_file to $pdf_file..."
            docker run --rm \
                -v "$(pwd):/root" \
                --entrypoint epstopdf \
                $IMAGE_NAME \
                "/root/$eps_file" --outfile="/root/$pdf_file"
        else
            echo "Skipping $eps_file (PDF is up to date)"
        fi
    fi
done

# Run pandoc in Docker container
echo "Running pandoc to generate report.pdf..."
docker run --rm \
    -v "$(pwd):/root" \
    --entrypoint pandoc \
    $IMAGE_NAME \
    /root/src/report.md --defaults /root/defaults.yaml -o /root/_build/report.pdf

if [ $? -eq 0 ]; then
    echo "Success! Report generated at _build/report.pdf"
else
    echo "Error: Failed to generate report."
    exit 1
fi

