import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.data_loader import load_raw_text, GPTDatasetV1
from src.tokenizer import GPT2Tokenizer
from src.model.baseline_lm import BaselineLM
from src.model.attention import CausalSelfAttention

def main():
    # load raw text

    raw_text=load_raw_text()
    print("Total characters: ",len(raw_text))


    #tokenizer
    tokenizer=GPT2Tokenizer()

    #encode text
    # token_ids=[]

    token_ids=tokenizer.encode(raw_text)
    print("Total tokens: ",len(token_ids))
    # print("Encoded: ",token_ids[:100])
    # decoded=tokenizer.decode(token_ids)
    # print("Decoded: ",decoded[:100])

    #creating dataset
    dataset=GPTDatasetV1(token_ids,max_length=64,stride=64)
    print("Total sequences: ",len(dataset))


    #inspecting batches

    dataloader= DataLoader(dataset,batch_size=4,shuffle=True)



    vocab_size=tokenizer.tokenizer.n_vocab
    model=BaselineLM(vocab_size)
    batch_x, batch_y = next(iter(dataloader))
    print(batch_x.shape)
    print(batch_y.shape)

    print(batch_x[0][:10])
    print(batch_y[0][:10])

    logits = model(batch_x)


    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        batch_y.view(-1)
    )
    print("Loss:", loss.item())

    dummy_x=torch.randn(4,64,128)
    attention=CausalSelfAttention(d_in=128,d_out=128,context_length=64)
    context=attention(dummy_x)
    print("Context shape:", context.shape)

if __name__=="__main__": 
    main()