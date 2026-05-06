# Use the official Jupyter base image
FROM jupyter/base-notebook:python-3.11

USER root
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_USER}

# 1. Install dependencies. 
# We add 'protobuf' which is often required for the T5 tokenizer (sentencepiece).
RUN mamba install -y --quiet \
    pytorch-cpu \
    transformers \
    sentencepiece \
    protobuf \
    -c pytorch -c huggingface -c conda-forge

# 2. Pre-cache the model weights using a more direct method.
# Instead of 'pipeline', we load the Auto classes. This avoids the "Task" registry error
# and ensures the tokenizer and model weights are both saved.
RUN python3 -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
    m_name = 'google/flan-t5-small'; \
    AutoTokenizer.from_pretrained(m_name); \
    AutoModelForSeq2SeqLM.from_pretrained(m_name)"

WORKDIR /home/jovyan
