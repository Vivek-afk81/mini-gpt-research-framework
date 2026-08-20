from pathlib import Path
import torch
from torch.utils.data import Dataset

def load_raw_text(data_dir="data"):
    data_path=Path(data_dir)
    text=[]

    for file_path in data_path.glob("*.txt"):
        with open(file_path,"r",encoding="utf-8") as f:
            text.append(f.read())
    return "\n\n".join(text)

class GPTDatasetV1(Dataset):
    def __init__(self,token_ids,max_length=64,stride=64):
        self.input_ids=[]
        self.target_ids=[]


        #use a sliding window ro chunk the text into overlapping sequences
        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk=token_ids[i:i+max_length]
            target_chunk=token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]
    
##A map-style dataset is one that implements the _getitem_() and len_() protocols, and represents a map from
#(possibly non-integral) indices/keys to data samples.


def create_datasets(token_ids, max_length=128, stride=128, test_ratio=0.1):
    """Split token_ids into train/test and return two GPTDatasetV1 instances."""
    split = int(len(token_ids) * (1 - test_ratio))
    train_ids = token_ids[:split]
    test_ids = token_ids[split:]
    train_dataset = GPTDatasetV1(train_ids, max_length=max_length, stride=stride)
    test_dataset = GPTDatasetV1(test_ids, max_length=max_length, stride=stride)
    return train_dataset, test_dataset