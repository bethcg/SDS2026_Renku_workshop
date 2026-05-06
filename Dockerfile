# Use the official Jupyter base image
FROM jupyter/base-notebook:python-3.11

USER root

# Install git for Renku lineage tracking
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_USER}

# 1. Use Mamba (standard in Jupyter images) to install essentials. 
# This is much faster and more reliable than pip in this specific base image.
RUN mamba install -y --quiet \
    pytorch-cpu \
    transformers \
    sentencepiece \
    -c pytorch -c huggingface -c conda-forge

# 2. Pre-cache the model weights
# We use the 'transformers' library to download the model into the container layers
RUN python3 -c "from transformers import pipeline; pipeline('summarization', model='google/flan-t5-small')"

# Set the working directory
WORKDIR /home/jovyan
