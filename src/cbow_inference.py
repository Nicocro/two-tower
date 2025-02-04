
import torch
import json
import torch.nn.functional as F

from src.cbow import CBOW
from src.tokenizer import W2VTokenizer


with open("text8_vocab.json", "r") as f:
    vocab = json.load(f)

tokenizer = W2VTokenizer(min_freq=7, use_stemming=False)
tokenizer.vocab = vocab
tokenizer.reverse_vocab = {v: k for k, v in vocab.items()}

VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 64

model = CBOW(VOCAB_SIZE, EMBEDDING_DIM)
model.load_state_dict(torch.load('./cbow_text8_weights.pt', map_location=torch.device('cpu')))
model.eval()

# testing the embeddings quality 
WORD = "hjhkjnkj"

word_tokens = tokenizer.tokenize(WORD)
word_tensor = torch.tensor(word_tokens).unsqueeze(0)
word_embedding = model.embeddings(word_tensor)

# Compute cosine similarity between the word embedding and all other embeddings
all_embeddings = model.embeddings.weight.data
cosine_similarities = F.cosine_similarity(word_embedding, all_embeddings.unsqueeze(0), dim=2)

# Get the top 5 most similar embeddings
top_k_similarities, top_k_indices = torch.topk(cosine_similarities, 5)

# Decode the indices back to words
similar_words = [tokenizer.reverse_vocab[idx.item()] for idx in top_k_indices[0]]

print(f"The 5 most similar words to '{WORD}' are: {similar_words}")