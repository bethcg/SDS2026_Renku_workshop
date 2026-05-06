import torch
import sys
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# 1. Configuration - Using the model pre-cached in your Docker image
MODEL_NAME = "google/flan-t5-small"

def run_summarization(input_file, output_file):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 2. Explicitly load the model and tokenizer
    # This bypasses the 'KeyError' by providing the objects directly to the pipeline
    print(f"Loading model: {MODEL_NAME}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. Create the pipeline with explicit task and components
    # We use 'text2text-generation' as it is the most stable task mapping for T5
    summarizer = pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer
    )

    # 4. Read the input text
    try:
        with open(input_file, 'r') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # 5. Perform Summarization
    # We add the 'summarize: ' prefix which T5 models expect for this task
    print("Generating summary...")
    input_text = f"summarize: {raw_text}"
    
    # max_length controls the 'TL;DR' nature of the output
    results = summarizer(input_text, max_length=50, min_length=10, do_sample=False)
    summary_text = results[0]['generated_text']

    # 6. Save the output
    with open(output_file, 'w') as f:
        f.write(summary_text)
    
    print(f"Success! Summary saved to {output_file}")

if __name__ == "__main__":
    # This structure allows participants to use 'renku run'
    if len(sys.argv) < 3:
        print("Usage: python src/summarize.py <input.txt> <output.txt>")
    else:
        run_summarization(sys.argv[1], sys.argv[2])
