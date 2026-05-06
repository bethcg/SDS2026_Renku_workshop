from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/flan-t5-small"
def download():
    print(f"Downloading {model_name}...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSeq2SeqLM.from_pretrained(model_name)

if __name__ == "__main__":
    download()
