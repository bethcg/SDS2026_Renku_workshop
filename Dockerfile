FROM jupyter/base-notebook:python-3.11

USER root
# Install git and essential C++ dependencies for high-performance tokenizers
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libstdc++6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_USER}

# 1. Install PyTorch CPU and the specific libraries needed for T5
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers==4.38.0 \
    sentencepiece \
    protobuf==3.20.*

# 2. Pre-cache the model weights
# We use AutoModelForSeq2SeqLM. This ensures weights are saved to /home/jovyan/.cache
RUN python3 -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
    m_name = 'google/flan-t5-small'; \
    AutoTokenizer.from_pretrained(m_name); \
    AutoModelForSeq2SeqLM.from_pretrained(m_name)"

# 3. Final Verification (NO PIPELINE)
# We test the model by doing a direct manual encoding/decoding. 
# This confirms the model is functional without triggering the buggy Pipeline Registry.
RUN python3 -c "import torch; from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
    m_name = 'google/flan-t5-small'; \
    tokenizer = AutoTokenizer.from_pretrained(m_name); \
    model = AutoModelForSeq2SeqLM.from_pretrained(m_name); \
    inputs = tokenizer('summarize: The build is working.', return_tensors='pt'); \
    outputs = model.generate(**inputs, max_new_tokens=10); \
    result = tokenizer.decode(outputs[0], skip_special_tokens=True); \
    print('NLP Model Verification Success:', result)"

WORKDIR /home/jovyan
