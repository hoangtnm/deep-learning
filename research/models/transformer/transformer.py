import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class TransformerModel(nn.Module):
    """Transformer model with necessary modules.

    Args:
        src_vocab: Number of tokens in source vocabulary.
        tgt_vocab: Number of tokens in target vocabulary.
        d_model: the number of expected features in the encoder/decoder inputs.
        nhead: the number of heads in the multi-head-attention models.
        num_encoder_layers: the number of sub-encoder-layers in the encoder.
        num_decoder_layers: the number of sub-decoder-layers in the decoder.
        dim_feedforward: the dimension of the feedforward network model.

    Examples:
        >>> model = TransformerModel(src_vocab, tgt_vocab, num_layers)
        >>> input_embed = model.input_embed
        >>> output_embed = model.output_embed
        >>> pos_encoder = model.pos_encoder
        >>> src = torch.LongTensor([[1,2,4,5],
                                    [4,3,2,9]])
        >>> src = input_embed(src).transpose(0, 1)
        >>> src = pos_encoder(src)
        >>> tgt = torch.LongTensor([[2,4,5,4],
                                    [3,2,9,8]])
        >>> tgt = output_embed(tgt).transpose(0, 1)
        >>> tgt = pos_encoder(tgt)
        >>> out = model(tgt)
    """

    def __init__(self,
                 src_vocab: int,
                 tgt_vocab: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = Transformer(d_model, nhead, num_encoder_layers,
                                       num_decoder_layers, dim_feedforward,
                                       dropout)
        self.input_embed = Embedding(src_vocab, d_model)
        self.output_embed = Embedding(tgt_vocab, d_model)
        self.generator = Generator(d_model, tgt_vocab)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None) -> Tensor:
        output = self.transformer(src, tgt, src_mask, tgt_mask)
        output = self.generator(output)
        return output


class Transformer(nn.Module):
    """A transformer model.

    References:
    - [The Annotated Transformer by harvardnlp](
        http://nlp.seas.harvard.edu/2018/04/03/attention.html
    )
    - [nn.Transformer by PyTorch](
        https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    )

    Args:
        d_model: the number of expected features in the encoder/decoder inputs.
        nhead: the number of heads in the multi-head-attention models.
        num_encoder_layers: the number of sub-encoder-layers in the encoder.
        num_decoder_layers: the number of sub-decoder-layers in the decoder.
        dim_feedforward: the dimension of the feedforward network model.
        dropout: the dropout value.

    Examples:
        >>> model = Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = model(src, tgt)
    """

    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1) -> None:
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        self._init_weights()

        self.d_model = d_model
        self.nhead = nhead

    def _init_weights(self):
        """"Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None) -> Tensor:
        """Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder.
            tgt: the sequence to the decoder.
            src_mask: the additive mask for the src sequence.
            tgt_mask: the additive mask for the tgt sequence.

        Returns:
            A decoded sequence Tensor.

        Shape:
            - src: (S, N, E).
            - tgt: (T, N, E).
            - src_mask: (S, S).
            - tgt_mask: (T, T).
            - output: (T, N, E).
        """
        memory = self.encoder(src, src_mask)
        return self.decoder(tgt, memory, tgt_mask, src_mask)

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """"Generate a square mask for the sequence.

        The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        # mask = torch.triu(torch.ones(sz, sz), diagonal=1).float() == 0
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N decoder layers.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer.
        num_layers: the number of sub-encoder-layers in the encoder.
    """

    def __init__(self, encoder_layer, num_layers: int):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(encoder_layer.d_model)
        self.num_layers = num_layers

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder.
            mask: the mask for the src sequence.

        Shape:
            - src: (S, N, E).
            - mask: (S, S).
        """
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        return self.norm(output)


class TransformerDecoder(nn.Module):
    """TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer.
        num_layers: the number of sub-decoder-layers in the decoder.
    """

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = nn.LayerNorm(decoder_layer.d_model)
        self.num_layers = num_layers

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder.
            memory: the sequnce from the last layer of the encoder.
            tgt_mask: the mask for the tgt sequence.
            memory_mask: the mask for the memory sequence.

        Shape:
            - tgt: (T, N, E).
            - memory: (S, N, E).
            - tgt_mask: (T, T).
            - memory_mask: (S, S).
        """
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)
        return self.norm(output)


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input.
        nhead: the number of heads in the multi-head-attention models.
        dim_feedforward: the dimension of the feedforward network model.
        dropout: the dropout value.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(nhead, d_model, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.d_model = d_model

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer.
            src_mask: the mask for the src sequence.

        Shape:
            - src: (S, N, E).
            - src_mask: (S, S).
        """
        # Multi-head self-attention
        src2 = self.self_attn(src, src, src, src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Position-wise Feed-Forward Networks
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input.
        nhead: the number of heads in the multi-head-attention models.
        dim_feedforward: the dimension of the feedforward network model.
        dropout: the dropout value.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(nhead, d_model, dropout)
        self.src_attn = MultiheadAttention(nhead, d_model, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.d_model = d_model

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer.
            memory: the sequnce from the last layer of the encoder.
            tgt_mask: the mask for the tgt sequence.
            memory_mask: the mask for the memory sequence.

        Shape:
            - tgt: (T, N, E).
            - memory: (S, N, E).
            - tgt_mask: (T, T).
            - memory_mask: (S, S).
        """
        # Multi-head self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Multi-head attention over the output of the encoder stack
        tgt2 = self.src_attn(tgt, memory, memory, memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Position-wise Feed-Forward Networks
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention. One head. One spatial dimension.

    See details at
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L6078
    """

    def __init__(self, embed_dim: int, attn_droput: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scaling = 1 / math.sqrt(embed_dim)
        self.dropout = nn.Dropout(attn_droput)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Shape:
            - Inputs:
            - q: (T, N, E)
            - k: (S, N, E)
            - v: (S, N, E)
            - attn_mask: 2D mask (T, S). attn_mask ensure that position i is
            allowed to attend the unmasked positions.
            - Outputs:
            - attn_output: (T, N, E)
            - attn_weights: (N, T, S)

            Note:
            T is the target sequence length
            S is the source sequence length
            N is the batch size
            E is the embedding dimension
        """
        tgt_len, batch_size, embed_dim = q.size()
        assert embed_dim == self.embed_dim
        assert q.size(0) == v.size(0) and k.size(1) == v.size(1)

        if attn_mask is not None:
            assert attn_mask.dim() == 2
            attn_mask = attn_mask.unsqueeze(0)

        q = q * self.scaling
        # Ensure q, k, v has shape (N, L, E)
        # L is the sequence length
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        src_len = k.size(1)

        logits = torch.bmm(q, k.transpose(-2, -1))
        assert list(logits.size()) == [batch_size, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                logits.masked_fill_(attn_mask, float('-inf'))
            else:
                logits += attn_mask

        attn_weights = self.dropout(F.softmax(logits, dim=-1))
        output = torch.bmm(attn_weights, v)
        assert list(output.size()) == [
            batch_size, tgt_len, self.embed_dim
        ]
        output = output.transpose(0, 1).contiguous().view(
            tgt_len, batch_size, embed_dim
        )
        return output, attn_weights


class MultiheadAttention(nn.Module):
    """Multi-Head Attention.

    It allows the model to jointly attend to information
    from different representation subspaces at different positions.

    References:
    - [MultiheadAttention by PyTorch](
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention
    )
    - [MultiHeadAttention by Yu-Hsiang Huang](
        https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/fec78a687210851f055f792d45300d27cc60ae41/transformer/SubLayers.py
    )
    """

    def __init__(self, nhead: int, embed_dim: int, dropout: float = 0.,
                 bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.nhead = nhead
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.head_dim = embed_dim // nhead
        assert self.head_dim * nhead == self.embed_dim, (
            'embed_dim must be divisible by nhead')

        self.bias_k = self.bias_v = None

        self.w_qs = nn.Linear(embed_dim, embed_dim, bias)
        self.w_ks = nn.Linear(embed_dim, self.kdim, bias)
        self.w_vs = nn.Linear(embed_dim, self.vdim, bias)
        self.fc = nn.Linear(embed_dim, embed_dim)

        self.scaling = 1 / math.sqrt(self.head_dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Shape:
            - Inputs:
            - q: (T, N, E)
            - k: (S, N, E)
            - v: (S, N, E)
            - attn_mask: 2D mask (T, S). attn_mask ensure that position i is
            allowed to attend the unmasked positions.
            - Outputs:
            - attn_output: (T, N, E)
            - attn_weights: (N, T, S)

            Note:
            - T is the target sequence length
            - S is the source sequence length
            - N is the batch size
            - E is the keembedding dimension
        """
        tgt_len, batch_size, embed_dim = q.size()
        src_len = k.size(0)
        assert embed_dim == self.embed_dim
        assert k.size(0) == v.size(0) and k.size(1) == v.size(1)

        # Linearly project the queries, keys and values h times
        q = self.w_qs(q).view(tgt_len, batch_size, self.nhead, self.head_dim)
        k = self.w_ks(k).view(src_len, batch_size, self.nhead, self.head_dim)
        v = self.w_vs(v).view(src_len, batch_size, self.nhead, self.head_dim)

        if attn_mask is not None:
            assert attn_mask.dtype in [torch.float32, torch.uint8]
            assert attn_mask.dim() == 2
            # Head axis broadcasting,
            # which lets same mask applied to all h heads.
            attn_mask = attn_mask.unsqueeze(0)

        # Multi-Head Attention
        # On each of these projected versions of queries, keys and values
        # we then perform the attention function in parallel.
        q = q * self.scaling
        q = q.contiguous().view(
            tgt_len, batch_size*self.nhead, self.head_dim
        ).transpose(0, 1)
        k = k.contiguous().view(
            -1, batch_size*self.nhead, self.head_dim
        ).transpose(0, 1)
        v = v.contiguous().view(
            -1, batch_size*self.nhead, self.head_dim
        ).transpose(0, 1)
        src_len = k.size(1)

        # q has shape (batch_size*nhead, tgt_len, head_dim)
        # k, v has shape (batch_size*nhead, src_len, head_dim)
        logits = torch.bmm(q, k.transpose(1, 2))
        assert list(logits.size()) == [batch_size*self.nhead, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                logits.masked_fill_(attn_mask, float('-inf'))
            else:
                logits += attn_mask

        weights = self.dropout(F.softmax(logits, dim=-1))
        output = torch.bmm(weights, v)
        assert list(output.size()) == [
            batch_size*self.nhead, tgt_len, self.head_dim]
        output = output.transpose(0, 1).contiguous().view(
            tgt_len, batch_size, embed_dim)
        output = self.fc(output)

        # Average attention weights over heads
        weights = weights.view(batch_size, self.nhead, tgt_len, src_len)
        weights = weights.sum(dim=1) / self.nhead

        return output, weights


class Generator(nn.Module):
    """Standard linear + softmax generation."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Embedding(nn.Module):
    def __init__(self, vocab: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        self.vocab = vocab

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        # Create embedding along each position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: the sequence fed to the positional encoder.

        Returns:
            Encoded sequence with the same shape.

        Shape:
            x: (L, batch_size, embed_dim)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# class MultiHeadAttention(nn.Module):
#     """Multi-Head Attention.

#     It allows the model to jointly attend to information
#     from different representation subspaces at different positions.

#     Args:
#         nhead: parallel attention heads.
#         d_model: total dimension of the model.
#         dropout: a Dropout layer on attn_output_weights. Default: 0.1.
#     """

#     def __init__(self, nhead: int, d_model: int, dropout=0.1):
#         super().__init__()
#         assert d_model % nhead == 0, 'd_model must be divisible by nhead'
#         self.d_k = d_model // nhead
#         self.nhead = nhead
#         self.linears = _get_clones(nn.Linear(d_model, d_model), 4)
#         self.w_qs = nn.Linear(d_model, d_model, bias=False)
#         self.w_ks = nn.Linear(d_model, d_model, bias=False)
#         self.w_vs = nn.Linear(d_model, d_model, bias=False)
#         self.fc = nn.Linear(d_model, d_model, bias=False)
#         self.attention = ScaledDotProductAttention()
#         # self.dropout = nn.Dropout(dropout)

#     def forward(self, q: Tensor, k: Tensor, v: Tensor, mask=None):
#         """Maps a query and a set of key-value pairs to an output.

#         Args:
#             q:
#             k:
#             v:
#             mask:

#         Returns:
#             attn_output: attention output.
#         """
#         batch_size = q.size(0)

#         # Do all the linear projections in batch from d_model => nhead x d_k
#         q, k, v = [
#             l(x).view(batch_size, -1, self.nhead, self.d_k)
#             for l, x in zip(self.linears, (q, k, v))
#         ]

#         if mask is not None:
#             # Same mask applied to all nhead heads.
#             mask = mask.unsqueeze(1)

#         # Apply attention on all the projected vectors in batch.
#         # x, self.atnn = attention(q, k, v, mask, self.dropout)
#         x, _ = self.attention(q, k, v, mask)

#         # Transpose x from (batch_size, seq_length, nhead, d_k)
#         # to (batch_size, nhead, seq_length, d_k) and
#         # concats using a view and apply a final linear:
#         # (batch_size, seq_length, nhead*d_k)
#         x = x.transpose(1, 2).contiguous()
#         x = x.view(batch_size, -1, self.nhead * self.d_k)
#         attn_output = self.linears[-1](x)
#         return attn_output
