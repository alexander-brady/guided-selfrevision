import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional


def convert_to_tensor(
    sequences: List[List[int]],
    pad_token_id: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert a list of sequences (lists of integers) to a padded tensor.
    
    Args:
        sequences (List[List[int]]): List of sequences to convert.
        pad_token_id (int): The token ID to use for padding.
        device (Optional[torch.device]): The device to place the tensor on.
        
    Returns:
        torch.Tensor: A padded tensor containing the sequences.
    """
    padded_sequences = pad_sequence(
        [torch.tensor(seq) for seq in sequences],
        batch_first=True,
        padding_value=pad_token_id,
    )
    
    if device is not None:
        padded_sequences = padded_sequences.to(device)
    
    return padded_sequences