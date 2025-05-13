# Build
docker build -t hemu-dev .

# Run with GPU support and mounted volumes
docker run --gpus all \
    -v /path/to/local/data:/app/data/Helio \
    -v /path/to/local/analysis:/app/analysis/runs \
    hemu-dev