FROM jupyter/base-notebook:python-3.11

USER root
# Install build essentials for tokenizers and compression tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    unrar \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_USER}

# 1. Install PyTorch CPU and NLP libraries
# Use --extra-index-url so pip can see both the Torch CPU repo AND the standard PyPI repo
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    transformers==4.38.0 \
    sentencepiece \
    protobuf==3.20.*

# 2. Pre-cache the model weights
RUN python3 -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
    m_name = 'google/flan-t5-small'; \
    AutoTokenizer.from_pretrained(m_name); \
    AutoModelForSeq2SeqLM.from_pretrained(m_name)"

# 3. Manual Verification (No Pipeline Registry dependency)
RUN python3 -c "import torch; from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
    m_name = 'google/flan-t5-small'; \
    tokenizer = AutoTokenizer.from_pretrained(m_name); \
    model = AutoModelForSeq2SeqLM.from_pretrained(m_name); \
    inputs = tokenizer('summarize: The build is finally correct.', return_tensors='pt'); \
    outputs = model.generate(**inputs, max_new_tokens=10); \
    result = tokenizer.decode(outputs[0], skip_special_tokens=True); \
    print('NLP Model Verification Success:', result)"

WORKDIR /home/jovyan
