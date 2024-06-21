import torch

class BPETokenizer:
    """ PyTorch-aware class that wraps the Encoder above """

    def __init__(self):
        self.encoder = get_encoder()

    def __call__(self, text, return_tensors='pt'):
        # PyTorch only; here because we want to match huggingface/transformers interface
        assert return_tensors == 'pt'
        # single string input for now, in the future potentially a list of strings
        assert isinstance(text, str)
        # encode and create a "batch dimension" of 1
        idx = [self.encoder.encode(text)]
        # wrap into PyTorch tensor
        out = torch.tensor(idx, dtype=torch.long)
        return out

    def decode(self, idx):
        # ensure a simple 1D tensor for now
        assert idx.ndim == 1
        # decode indices to text
        text = self.encoder.decode(idx.tolist())
        return text