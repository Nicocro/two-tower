import torch
import torch.nn.functional as F
import faiss
from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

# Import from parent directory (adjust imports based on your structure)
from src.model import QueryTower, PassageTower
from src.cbow import CBOW
from src.tokenizer import W2VTokenizer
from src.dataset import TTDataset

class SearchService():
    def __init__(self):
        # initialize models and index
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.query_tower = None
        self.passage_tower = None
        self.tokenizer = None
        self.index = None
        self.passage_ids = []

    def load_models(self, 
                   vocab_path: str = "text8_vocab.json",
                   cbow_weights_path: str = "cbow_text8_weights.pt",
                   query_weights_path: str = "weights/trained_query_tower.pt",
                   passage_weights_path: str = "weights/trained_passage_tower.pt"):
        """ Load all the models and weights"""
        self.tokenizer = W2VTokenizer()
        self.tokenizer.load_vocab(vocab_path)

        # Load tokenizer and CBOW
        self.tokenizer = W2VTokenizer()
        self.tokenizer.load_vocab(vocab_path)
        
        # Initialize CBOW
        cbow_model = CBOW(len(self.tokenizer.vocab), 64)
        cbow_model.load_state_dict(torch.load(cbow_weights_path, map_location=self.device))
        
        # Initialize towers
        self.query_tower = QueryTower(cbow_model).to(self.device)
        self.passage_tower = QueryTower(cbow_model).to(self.device)
        
        # Load trained weights
        self.query_tower.load_state_dict(torch.load(query_weights_path, map_location=self.device))
        self.passage_tower.load_state_dict(torch.load(passage_weights_path, map_location=self.device))
        
        # Set to eval mode
        self.query_tower.eval()
        self.passage_tower.eval()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(64)  # 64 is embedding dimension

    def build_passage_cache(self, dataset: TTDataset, batch_size: int = 32):
        """Create and store passage embeddings in FAISS."""
        if not self.passage_tower:
            raise RuntimeError("Models not loaded. Call load_models() first.")
            
        print("Building passage cache...")
        self.passage_tower.eval()
        
        passages_seen = set()
        all_embeddings = []
        
        for idx in tqdm(range(len(dataset))):
            item = dataset[idx]
            passages = [item['positive'], item['negative']]
            
            for passage in passages:
                if passage not in passages_seen:
                    # Tokenize passage
                    token_ids = torch.tensor(self.tokenizer.tokenize(passage)).unsqueeze(0).to(self.device)
                    length = torch.tensor([len(token_ids[0])]).to(self.device)
                    
                    with torch.no_grad():
                        embedding = self.passage_tower(token_ids, length)
                        # Normalize for cosine similarity
                        embedding = F.normalize(embedding, p=2, dim=1)
                    
                    all_embeddings.append(embedding.cpu().numpy())
                    self.passage_ids.append(passage)
                    passages_seen.add(passage)
                    
                    # Add to FAISS index in batches
                    if len(all_embeddings) >= batch_size:
                        embeddings_batch = np.vstack(all_embeddings)
                        self.index.add(embeddings_batch)
                        all_embeddings = []
        
        # Add remaining embeddings
        if all_embeddings:
            embeddings_batch = np.vstack(all_embeddings)
            self.index.add(embeddings_batch)
            
        print(f"Cached {len(self.passage_ids)} unique passages")


    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for most similar passages to query."""
        if not self.query_tower or not self.index:
            raise RuntimeError("Models not loaded or passages not cached. Initialize properly first.")
            
        # Tokenize query
        token_ids = torch.tensor(self.tokenizer.tokenize(query)).unsqueeze(0).to(self.device)
        length = torch.tensor([len(token_ids[0])]).to(self.device)
        
        # Get query embedding
        with torch.no_grad():
            query_embedding = self.query_tower(token_ids, length)
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
        
        # Search in index
        query_numpy = query_embedding.cpu().numpy()
        scores, indices = self.index.search(query_numpy, k)
        
        # Return results
        return [
            (self.passage_ids[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
        ]
    
    # Simple usage example
if __name__ == "__main__":
    # Initialize service
    service = SearchService()
    
    # Load models
    service.load_models()
    
    # Load dataset and build cache
    ds = load_dataset("microsoft/ms_marco", "v1.1")
    test_dataset = TTDataset(ds, split='test')
    service.build_passage_cache(test_dataset)
    
    # Test search
    results = service.search("what is machine learning?")
    for passage, score in results:
        print(f"Score: {score:.3f}")
        print(f"Passage: {passage[:200]}...\n")