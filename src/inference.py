import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List, Tuple
import numpy as np
import faiss
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Import your custom classes (adjust paths as needed)
from src.model import QueryTower, CBOW
from src.dataset import TTDataset
from src.tokenizer import W2VTokenizer

def get_device():
    # Use GPU if available; else MPS (for Apple Silicon) if available; else CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
        
class SearchEngine:
    def __init__(self, query_tower: QueryTower, passage_tower: QueryTower, tokenizer: W2VTokenizer, device=get_device()):
        self.device = device
        print(f"Initializing SearchEngine on {self.device}")
        self.query_tower = query_tower.to(self.device)
        self.passage_tower = passage_tower.to(self.device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize FAISS index
        self.embedding_dim = 64  # Your embedding dimension
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product = cosine similarity for normalized vectors
        self.passage_ids = []  # Store passage texts
        
    def build_passage_cache(self, dataset: TTDataset, batch_size: int = 32):
        """Create and store passage embeddings in FAISS."""
        self.query_tower.eval()
        self.passage_tower.eval()
        
        passages_seen = set()
        all_embeddings = []
        
        for idx in tqdm(range(len(dataset)), desc="Building passage cache"):
            item = dataset[idx]
            passages = [item['positive'], item['negative']]
            
            for passage in passages:
                if passage not in passages_seen:
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
        """Search using FAISS index."""
        self.query_tower.eval()
        
        token_ids = torch.tensor(self.tokenizer.tokenize(query)).unsqueeze(0).to(self.device)
        length = torch.tensor([len(token_ids[0])]).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.query_tower(token_ids, length)
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
        
        # Search in FAISS index
        query_numpy = query_embedding.cpu().numpy()
        scores, indices = self.index.search(query_numpy, k)
        
        return [
            (self.passage_ids[idx], score)
            for idx, score in zip(indices[0], scores[0])
        ]

def setup_prerequisites():
    # Initialize tokenizer
    tokenizer = W2VTokenizer()
    tokenizer.load_vocab("text8_vocab.json")
    
    # Get vocab size from tokenizer
    VOCAB_SIZE = len(tokenizer.vocab)
    EMBEDDING_DIM = 64
    
    # Initialize and load pretrained CBOW
    cbow_model = CBOW(VOCAB_SIZE, EMBEDDING_DIM)
    cbow_model.load_state_dict(
        torch.load('./cbow_text8_weights.pt', map_location=torch.device('cpu'))
    )
    
    return cbow_model, tokenizer

def evaluate(search_engine: SearchEngine, test_dataset: TTDataset, k: int = 5):
    """Compute MRR and MAP metrics."""
    mrr_sum = 0
    ap_sum = 0
    
    for idx in tqdm(range(len(test_dataset)), desc="Evaluating"):
        item = test_dataset[idx]
        query = item['query']
        relevant_passage = item['positive']
        
        results = search_engine.search(query, k)
        
        # Compute metrics
        for rank, (passage, _) in enumerate(results, 1):
            if passage == relevant_passage:
                mrr_sum += 1.0 / rank
                ap_sum += 1.0 / rank
                break
    
    num_queries = len(test_dataset)
    mrr = mrr_sum / num_queries
    map_score = ap_sum / num_queries
    
    return {
        'MRR': mrr,
        'MAP': map_score
    }

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load prerequisites
    print("Loading prerequisites...")
    cbow_model, tokenizer = setup_prerequisites()
    
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("microsoft/ms_marco", "v1.1")
    test_dataset = TTDataset(ds, split='test')
    
    # Initialize models
    print("Initializing models...")
    query_tower = QueryTower(embedder=cbow_model)
    passage_tower = QueryTower(embedder=cbow_model)
    
    # Load trained weights
    print("Loading model weights...")
    query_tower.load_state_dict(torch.load('weights/trained_query_tower.pt', map_location=torch.device('cpu')))
    passage_tower.load_state_dict(torch.load('weights/trained_passage_tower.pt', map_location=torch.device('cpu')))
    
    # Create search engine
    search_engine = SearchEngine(query_tower, passage_tower, tokenizer, device)
    
    # Build passage cache
    print("Building passage cache...")
    search_engine.build_passage_cache(test_dataset)
    
    # Evaluate
    print("Evaluating model...")
    metrics = evaluate(search_engine, test_dataset)
    print(f"MRR@5: {metrics['MRR']:.3f}")
    print(f"MAP@5: {metrics['MAP']:.3f}")
    
    # Interactive search demo
    while True:
        query = input("\nEnter a search query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        results = search_engine.search(query)
        print("\nSearch results:")
        for i, (passage, score) in enumerate(results, 1):
            print(f"\nRank {i} (score: {score:.3f}):")
            print(f"Passage: {passage[:200]}...")


main()