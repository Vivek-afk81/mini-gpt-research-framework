import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.dataset import load_raw_text, GPTDatasetV1
from src.tokenizer import GPT2Tokenizer
from src.model.gpt import GPT

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


    #inspecting batches

    dataloader= DataLoader(dataset,batch_size=4,shuffle=True)



    vocab_size=tokenizer.tokenizer.n_vocab
    model=GPT(vocab_size)
    
    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    #TRAINING LOOP

    num_epochs=5
    for epoch in range(num_epochs):
        total_loss=0

        for batch_x,batch_y in dataloader:
            optimizer.zero_grad()
            logits=model(batch_x)

            loss=F.cross_entropy(
                logits.view(-1,vocab_size),
                batch_y.view(-1)
            )
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        avg_loss=total_loss/len(dataloader)
        print(f"Epoch {epoch+1}: {avg_loss:.4f}")





if __name__=="__main__": 
    main()