import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from typing import List, Dict
import wandb
from tqdm.auto import tqdm

# Import all necessary components
from src.cbow import CBOW
from src.tokenizer import W2VTokenizer
from src.dataset import TTDataset
from src.model import QueryTower, PassageTower, TripletLoss

def get_device():
    # Use GPU if available; else MPS (for Apple Silicon) if available; else CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def setup_prerequisites():
    # Initialize tokenizer
    tokenizer = W2VTokenizer()
    tokenizer.load_vocab("text8_vocab.json")  # Make sure this path is correct
    
    # Get vocab size from tokenizer
    VOCAB_SIZE = len(tokenizer.vocab)
    EMBEDDING_DIM = 64  # As defined in your original training
    
    # Initialize and load pretrained CBOW
    cbow_model = CBOW(VOCAB_SIZE, EMBEDDING_DIM)
    cbow_model.load_state_dict(
        torch.load('./cbow_text8_weights.pt', map_location=torch.device('cpu'))
    )
    
    return cbow_model, tokenizer


def collate_tower_batch(tokenizer: W2VTokenizer, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for the TTDataset that handles tokenization and padding
    """

    queries = [item['query'] for item in batch]
    positives = [item['positive'] for item in batch]
    negatives = [item['negative'] for item in batch]
    
    # Tokenize all texts
    query_tokens = [tokenizer.tokenize(q) for q in queries]
    pos_tokens = [tokenizer.tokenize(p) for p in positives]
    neg_tokens = [tokenizer.tokenize(n) for n in negatives]
    
    # Get max lengths
    max_query_len = max(len(t) for t in query_tokens)
    max_pos_len = max(len(t) for t in pos_tokens)
    max_neg_len = max(len(t) for t in neg_tokens)
    
    # Pad sequences
    pad_id = tokenizer.special_tokens['<PAD>']
    
    padded_queries = [
        t + [pad_id] * (max_query_len - len(t)) for t in query_tokens
    ]
    padded_positives = [
        t + [pad_id] * (max_pos_len - len(t)) for t in pos_tokens
    ]
    padded_negatives = [
        t + [pad_id] * (max_neg_len - len(t)) for t in neg_tokens
    ]
    
    # Convert to tensors
    return {
        'query_ids': torch.tensor(padded_queries),
        'query_lengths': torch.tensor([len(t) for t in query_tokens]),
        'positive_ids': torch.tensor(padded_positives),
        'positive_lengths': torch.tensor([len(t) for t in pos_tokens]),
        'negative_ids': torch.tensor(padded_negatives),
        'negative_lengths': torch.tensor([len(t) for t in neg_tokens])
    }

# In tt_train.py, modify the DataLoader creation:


def train_model(query_tower: nn.Module, 
                passage_tower: nn.Module, 
                train_loader: DataLoader,
                num_epochs: int = 3,
                learning_rate: float = 1e-3):
    
    device = get_device()
    query_tower.to(device)
    passage_tower.to(device)
    
    # Initialize wandb
    wandb.init(
        project="two-tower-search",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size
        }
    )

    # Set models to training mode
    query_tower.train()
    passage_tower.train()
    
    # Initialize loss and optimizer
    criterion = TripletLoss().to(device)
    params = list(filter(lambda p: p.requires_grad, query_tower.parameters())) + \
         list(filter(lambda p: p.requires_grad, passage_tower.parameters()))
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Get query and passage vectors
            query_vec = query_tower(batch['query_ids'], batch['query_lengths'])
            pos_vec = passage_tower(batch['positive_ids'], batch['positive_lengths'])
            neg_vec = passage_tower(batch['negative_ids'], batch['negative_lengths'])
            
            # Compute loss
            loss = criterion(query_vec, pos_vec, neg_vec)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            current_loss = loss.item()
            total_loss += loss.item()
            num_batches += 1

            # Update the progress bar postfix with the current batch loss
            pbar.set_postfix({'batch_loss': f'{current_loss:.4f}'})
            
            # Log to wandb per batch
            wandb.log({
                "batch_loss": current_loss,
                "batch": num_batches + epoch * len(train_loader)
            })
        
        # Print epoch statistics
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

        # Log to wandb per epoch
        wandb.log({
            "epoch": epoch,
            "avg_loss": avg_loss,
        })

    # Close wandb run
    wandb.finish()
    
    return query_tower, passage_tower


###usage

def main():

    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    # Setup all prerequisites
    cbow_model, tokenizer = setup_prerequisites()
    
    # Initialize two towers with CBOW
    query_tower = QueryTower(embedder=cbow_model)
    passage_tower = PassageTower(embedder=cbow_model)
    
    # Load MS MARCO dataset
    ds = load_dataset("microsoft/ms_marco", "v1.1")
    
    # Create dataset and dataloader
    dataset = TTDataset(ds, split='train')
    train_loader = train_loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        collate_fn=lambda b: collate_tower_batch(tokenizer, b)
    )
    
    # Train models
    trained_query_tower, trained_passage_tower = train_model(
        query_tower=query_tower,
        passage_tower=passage_tower,
        train_loader=train_loader,
        num_epochs=10,
        learning_rate=1e-4
    )
    
    # Optionally save the trained models
    torch.save(trained_query_tower.state_dict(), 'weights/trained_query_tower.pt')
    torch.save(trained_passage_tower.state_dict(), 'weights/trained_passage_tower.pt')


main()