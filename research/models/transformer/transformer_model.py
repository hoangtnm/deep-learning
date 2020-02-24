import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tokenizers import BertWordPieceTokenizer


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layer=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask, tgt_mask):
        pass


def get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot-Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    prob_attn = F.log_softmax(scores, dim=-1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(scores, value), prob_attn


class MultiHeadAttention(nn.Module):
    """Allows the model to jointly attend to information
    from different representation subspaces.

    Args:
        nhead: parallel attention heads.
        d_model: total dimension of the model.
        dropout: a Dropout layer on attn_output_weights. Default: 0.1.
    """

    def __init__(self, nhead, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.linears = get_clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """Maps a query and a set of key-value pairs to an output.

        Args:
            query (torch.Tensor): [description]
            key (torch.Tensor): [description]
            value (torch.Tensor): [description]
            mask (optional): [description]. Defaults to None.

        Returns:
            attn_output: attention output.
        """
        if mask is not None:
            # Same mask applied to all nhead heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.nhead, self.d_k)
                             for l, x in zip(self.linears, (query, key, value))]

        # Apply attention on all the projected vectors in batch.
        x, self.atnn = attention(query, key, value, mask, self.dropout)

        # Transpose x from (batch_size, seq_length, nhead, d_k)
        # to (batch_size, nhead, seq_length, d_k) and
        # concats using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous()
        x = x.view(nbatches, -1, self.nhead * self.d_k)
        attn_output = self.linears[-1](x)
        return attn_output
