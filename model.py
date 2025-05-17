import torch # type: ignore
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer once on import
best_model_loaded = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
best_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model_loaded.to(device)

# Translation function
def translate(text):
    input_ids = best_tokenizer.encode(text, return_tensors="pt").to(device)
    output = best_model_loaded.generate(input_ids)
    return best_tokenizer.decode(output[0], skip_special_tokens=True)
