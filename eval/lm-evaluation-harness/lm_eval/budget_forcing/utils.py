import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional


def convert_to_tensor(
    sequences: List[List[int]],
    pad_token_id: int,
    max_tokens: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert a list of sequences (lists of integers) to a padded tensor.
    
    Args:
        sequences (List[List[int]]): List of sequences to convert.
        pad_token_id (int): The token ID to use for padding.
        max_tokens (int): The maximum number of tokens in the sequences. 
            If a sequence exceeds this length, it will be left truncated.
        device (Optional[torch.device]): The device to place the tensor on.
        
    Returns:
        torch.Tensor: A padded tensor containing the sequences.
    """
    padded_sequences = pad_sequence(
        [torch.tensor(seq) for seq in sequences],
        batch_first=True,
        padding_value=pad_token_id,
    )

    padded_sequences = padded_sequences[:, -max_tokens:] 
    attention_mask = (padded_sequences != pad_token_id).long()
    
    if device is not None:
        padded_sequences = padded_sequences.to(device)
        attention_mask = attention_mask.to(device)

    return padded_sequences, attention_mask