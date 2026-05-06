# Use a Python 3.11 base image from Jupyter
FROM jupyter/base-notebook:python-3.11

USER root
# Install git for Renku tracking and basic build tools
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_USER}

# 1. Install NLP and Deep Learning essentials
# We include 'protobuf' as it is a strict requirement for the T5/SentencePiece tokenizer
RUN mamba install -y --quiet \
    pytorch-cpu \
    transformers \
    sentencepiece \
    protobuf \
    -c pytorch -c huggingface -c conda-forge

# 2. Pre-cache the model weights 
# We load the AutoModel and AutoTokenizer explicitly.
# This ensures both the model architecture and the vocabulary are baked into the image.
RUN python3 -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
    model_name = 'google/flan-t5-small'; \
    AutoTokenizer.from_pretrained(model_name); \
    AutoModelForSeq2SeqLM.from_pretrained(model_name)"

# 3. Final verification
# This confirms the 'summarization' task is registered and working before finishing the build
RUN python3 -c "from transformers import pipeline; \
    summarizer = pipeline('summarization', model='google/flan-t5-small'); \
    print('NLP Model Ready:', summarizer('Testing the summarization task registry.', max_length=10))"

WORKDIR /home/jovyan
