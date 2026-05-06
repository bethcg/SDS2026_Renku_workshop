# Use a minimal Jupyter base image with Python 3.10+
FROM jupyter/base-notebook:python-3.11

# Set labels for Renku compatibility (optional but recommended)
LABEL maintainer="Renku Workshop"

USER root
# Install git so Renku can track changes inside the container
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_USER}

# Install only the absolute essentials for the GenAI thread
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers \
    sentencepiece

# Pre-cache the flan-t5-small model weights
# This is the "Generative AI setup" part: baking the model into the image
RUN python3 -c "from transformers import pipeline; pipeline('summarization', model='google/flan-t5-small')"

# Set the working directory
WORKDIR /home/jovyan
