from transformers import AutoTokenizer

model = input("Enter model name: ")
text = input("Enter text to tokenize: ")

tokenizer = AutoTokenizer.from_pretrained(model)

tokens = tokenizer.tokenize(text)
ids = tokenizer(text).input_ids

print("\nTokens:", tokens)
print("Token IDs:", ids)
print("Vocabulary Size:", len(tokenizer))