import torch
from torch.utils.data import DataLoader
from src.data_loader import load_raw_text, GPTDatasetV1
from src.tokenizer import GPT2Tokenizer

def main():
    # load raw text

    raw_text=load_raw_text()
    print("Total characters: ",len(raw_text))

    #tokenizer
    tokenizer=GPT2Tokenizer()

    #encode text

    token_ids=tokenizer.encode(raw_text)
    print("Total tokens: ",len(token_ids))

    #creating dataset
    dataset=GPTDatasetV1(token_ids,max_length=64,stride=64)
    print("Total sequences: ",len(dataset))

    x,y=dataset[0]
    print("\nInput shape:", x.shape)
    print("\nTarget shape:", y.shape)

    print("\nDecoded Input:")
    print(tokenizer.decode(x.tolist()))

    print("\nDecoded Target:")
    print(tokenizer.decode(y.tolist()))


if __name__=="__main__": 
    main()