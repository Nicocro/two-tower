from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

import torch.nn.functional as F

from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

from src.dataset import TTDataset
from src.deploy.app.search import SearchService

# Define request/response models
class SearchRequest(BaseModel):
    query: str
    k: int = 5  # default to top 5 results

class SearchResponse(BaseModel):
    results: List[Dict[str, str | float]]

# Initialize FastAPI app
app = FastAPI(title="Search API")

# Initialize search service
search_service = SearchService()

@app.on_event("startup")
async def startup_event():
    """Initialize models and build cache on startup."""
    try:
        search_service.load_models()
        # Load dataset and build cache
        ds = load_dataset("microsoft/ms_marco", "v1.1")
        test_dataset = TTDataset(ds, split='test')
        search_service.build_passage_cache(test_dataset)
    except Exception as e:
        print(f"Error during initialization: {e}")
        raise

@app.post("/search", response_model=SearchResponse)
async def search_api(request: SearchRequest):
    """Search endpoint."""
    try:
        results = search_service.search(request.query, k=request.k)
        return SearchResponse(
            results=[
                {
                    "passage": passage,
                    "score": float(score)
                }
                for passage, score in results
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("src.deploy.app.main:app", host="0.0.0.0", port=8000, reload=True)