import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        combined = positive+negative
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        vocabulary = sorted({word for sentence in combined for word in sentence.split()})
        word_to_idx={word:idx+1 for idx,word in enumerate(vocabulary)}
        # 2. Encode each sentence by replacing words with their IDs
        encoded = [torch.tensor([word_to_idx[w]for w in s.split()])for s in combined]
        # 3. Combine positive + negative into one list of tensors
        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)
        
