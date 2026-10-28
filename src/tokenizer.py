import tiktoken


class GPT2Tokenizer:
    """
    Wrapper around tiktoken's GPT-2 BPE tokenizer.

    Provides a clean interface for encoding/decoding text and
    exposes vocab_size and eos_token_id for the model and dataset.
    """

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = self.tokenizer.encode(
            self.eos_token, allowed_special={"<|endoftext|>"}
        )[0]

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def encode(self, text, add_eos=False):
        """Encode text to token IDs. Optionally append EOS."""
        token_ids = self.tokenizer.encode(
            text, allowed_special={"<|endoftext|>"}
        )
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids):
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids)
