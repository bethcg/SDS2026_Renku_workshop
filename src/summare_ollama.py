import ollama
import sys
import os

def summarize_text(input_path, output_path):
    # 1. Read the input document
    with open(input_path, 'r') as f:
        document_text = f.read()

    # 2. Define the Summarization Instructions
    # Using a structured prompt ensures the model stays on task.
    prompt = f"""
    [INST] Summarize the following medical research text. 
    Focus on the objective, methodology, and key results. 
    Keep the summary under 100 words. [/INST]
    
    TEXT:
    {document_text}
    """

    # 3. Call the Local Model
    print(f"Summarizing {input_path} using local LLM...")
    response = ollama.generate(model='phi3:mini', prompt=prompt)
    summary = response['response']

    # 4. Save the Output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(summary)
    
    print(f"Summary successfully saved to {output_path}")

if __name__ == "__main__":
    # Allow command line arguments for Renku tracking
    if len(sys.argv) < 3:
        print("Usage: python summarize_ollama.py <input_file> <output_file>")
    else:
        summarize_text(sys.argv[1], sys.argv[2])
