# choose the version of CUDA and cuDNN that fits your needs
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 


# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY environment.yml .

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment
RUN conda env create -f environment.yml

# Copy source code
COPY configs ./configs
COPY models ./models
COPY metrics ./metrics
COPY train_and_eval ./train_and_eval
COPY utils ./utils

# provided as external volume
# COPY analysis ./analysis 
# COPY data ./data


# Set up entry point to activate conda environment
SHELL ["conda", "run", "-n", "HeMu-dev", "/bin/bash", "-c"]

# Default command
CMD ["python", "train_and_eval/regression_training_transf.py"]

# Add volume mount points for external data
VOLUME ["/app/data/Helio", "/app/analysis/runs"]