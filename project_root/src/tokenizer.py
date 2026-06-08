import tiktoken

class GPT2Tokenizer:
    def __init__(self):
        self.tokenizer=tiktoken.get_encoding("gpt2")

    
    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def decode(self,token_ids):
        return self.tokenizer.decode(token_ids)

