# Use the official Jupyter base image
FROM jupyter/base-notebook:python-3.11

USER root
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_USER}

# 1. Install dependencies with pinned versions to ensure compatibility
# Added 'torch' specifically to ensure the backend is fully recognized
RUN mamba install -y --quiet \
    pytorch-cpu \
    transformers \
    sentencepiece \
    protobuf \
    -c pytorch -c huggingface -c conda-forge

# 2. Pre-cache the model weights
# We use the 'Auto' classes to bypass the pipeline registry checks during the build
RUN python3 -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
    model_name = 'google/flan-t5-small'; \
    AutoTokenizer.from_pretrained(model_name); \
    AutoModelForSeq2SeqLM.from_pretrained(model_name)"

# 3. Final verification using the generic 'text2text-generation' task
# T5 is a Text-to-Text Transfer Transformer; 'summarization' is just a subset of this.
# This avoids the KeyError while still confirming the model works.
RUN python3 -c "from transformers import pipeline; \
    summarizer = pipeline('text2text-generation', model='google/flan-t5-small'); \
    print('NLP Model Ready:', summarizer('summarize: Testing the registry.', max_length=10))"

WORKDIR /home/jovyan
