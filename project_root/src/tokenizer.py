# import tiktoken

# class GPT2Tokenizer:
#     def __init__(self):
#         self.tokenizer=tiktoken.get_encoding("gpt2")
    
#     def encode(self, text):
#         return self.tokenizer.encode(text)
    
#     def decode(self,token_ids):
#         return self.tokenizer.decode(token_ids)

import tiktoken
from typing import List


class GPT2Tokenizer:
    """
    Lightweight wrapper around OpenAI's GPT-2 tokenizer.
    """

    def __init__(self):
        # Initialize tokenizer once
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> List[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        if not isinstance(token_ids, (list, tuple)):
            raise TypeError("token_ids must be a list or tuple of integers")

        return self.tokenizer.decode(token_ids)