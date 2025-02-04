import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random
import pandas as pd
from typing import Tuple, Dict



ds = load_dataset("microsoft/ms_marco", "v1.1")

class TTDataset(Dataset):
    def __init__(self, dataset: Dataset, split: str = "train"):
        self.ds_df = dataset[split].to_pandas()
        self.expl_df, self.expl_sel_df = self._preprocessing(self.ds_df)

    def _preprocessing(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Unroll the passages column and explode the is_selected and passage_text columns """

        passages_df = df["passages"].apply(pd.Series)
        df_unrolled = pd.concat([self.ds_df.drop(columns=["passages"]), passages_df], axis=1)
        df_unrolled.drop(columns=["answers", "url", "query_type", "wellFormedAnswers"], inplace=True)
        df_exploded = df_unrolled.explode(["is_selected", "passage_text"], ignore_index=True)
        df_expl_selected = df_exploded[df_exploded["is_selected"] == 1].copy()

        return df_exploded, df_expl_selected

    def __len__(self):
        """Return the number of unique queries in the dataset."""
        return len(self.expl_sel_df)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.expl_sel_df.iloc[idx]
        query = row['query']
        query_id = row['query_id']
        pos_passage = row['passage_text']

        while True:
            neg_idx = random.randint(0, len(self.expl_df) - 1)
            neg_row = self.expl_df.iloc[neg_idx]
            neg_passage = neg_row['passage_text']
            if neg_row['query_id'] != query_id:
                break
        
        return {
            'query': query,
            'positive': pos_passage,
            'negative': neg_passage
        }


## Test the dataset class

def test_dataset():
    # 1. Create a small instance
    ds = load_dataset("microsoft/ms_marco", "v1.1")
    dataset = TTDataset(ds, split='train')
    
    print(f"\n1. Dataset size: {len(dataset)}")
    
    # 2. Test single item retrieval
    item = dataset[0]
    print("\n2. First item structure:")
    print(f"Query: {item['query'][:100]}...")  # Print first 100 chars
    print(f"Positive: {item['positive'][:100]}...")
    print(f"Negative: {item['negative'][:100]}...")
    
    # 3. Test batch retrieval with DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    
    print("\n3. First batch structure:")
    print(f"Batch size: {len(batch)}")  # This will be the number of items in the batch dictionary
    for key in batch:
        print(f"Shape of {key}: {len(batch[key])}")  # Print shape of each tensor in batch
    print(f"First query in batch: {batch['query'][0][:100]}...")
    
    # 4. Verify negative sampling
    print("\n4. Testing negative sampling:")
    item1 = dataset[0]
    item2 = dataset[0]  # Get same index twice
    print(f"Same query, different negatives match?: {item1['negative'] == item2['negative']}")
    
    # 5. Check memory usage
    import sys
    print(f"\n5. Approximate memory usage: {sys.getsizeof(dataset.expl_df) / 1024 / 1024:.2f} MB")

    # 6. Additional checks
    print("\n6. Sample of different queries:")
    for i in range(3):
        sample = dataset[random.randint(0, len(dataset)-1)]
        print(f"\nSample {i+1}:")
        print(f"Query: {sample['query'][:50]}...")
        print(f"Positive: {sample['positive'][:50]}...")
        print(f"Negative: {sample['negative'][:50]}...")


