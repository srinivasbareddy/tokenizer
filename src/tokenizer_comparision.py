from transformers import AutoTokenizer

models = [
    "bert-base-cased",
    "bert-base-uncased",
    "gpt2",
    "google/flan-t5-small",
    "microsoft/Phi-3-mini-4k-instruct"
]

sentence = "Testing tokenizer implementation for AI models"

for model in models:
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokens = tokenizer.tokenize(sentence)

    print("\nModel:", model)
    print("Tokens:", tokens)
    print("Token Count:", len(tokens))