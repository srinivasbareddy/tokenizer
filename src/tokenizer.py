import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer

# define the sentence to tokenize
sentence = "Hello World!"

# load the pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# apply the tokenizer to the sentence and extract the token ids
token_ids = tokenizer(sentence).input_ids

print(token_ids)

for id in token_ids:
    print(tokenizer.decode(id))