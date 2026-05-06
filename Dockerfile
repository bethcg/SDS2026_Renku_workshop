# Use the official Jupyter base image
FROM jupyter/base-notebook:python-3.11

USER root
# Install git and essential C++ build tools for tokenizers
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_USER}

# 1. Install PyTorch CPU first and alone. 
# This ensures the backend is established before transformers tries to register tasks.
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install NLP libraries
RUN pip install --no-cache-dir \
    transformers \
    sentencepiece \
    protobuf

# 3. Pre-cache weights and FORCE task registration
# We use AutoModel classes to download the files.
# Then we run a dummy inference using the 'text-generation' task 
# (which your error log shows IS available) to verify the pipeline.
RUN python3 -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
    model_name = 'google/flan-t5-small'; \
    AutoTokenizer.from_pretrained(model_name); \
    AutoModelForSeq2SeqLM.from_pretrained(model_name)"

# 4. Final verification using 'text-generation' (which is active in your log)
# We map the Seq2Seq model to a generic pipeline to confirm the backend is alive.
RUN python3 -c "from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer; \
    model_name = 'google/flan-t5-small'; \
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name); \
    tokenizer = AutoTokenizer.from_pretrained(model_name); \
    pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer); \
    print('NLP Model Ready:', pipe('summarize: Testing.', max_length=10))"

WORKDIR /home/jovyan
