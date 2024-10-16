import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, embedding_size,
                 src_vocab_size, trg_vocab_size,
                 src_pad_idx,
                 num_heads,
                 num_encoder_layers,
                 num_decoder_layers,
                 forward_expansion,
                 dropout,
                 max_len,
                 device):

        super().__init__()

        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)

        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.src_pad_idx = src_pad_idx

        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        # src.shape: (src_len, N)

        src_mask = src.transpose(0, 1) == self.src_pad_idx
        # src_mask.shape: (N, src_len)

        return src_mask

    def forward(self, src, trg):
        src_seq_len, N = src.shape
        trg_seq_len, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N).to(self.device)
        )
        trg_positions = (
            torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, N).to(self.device)
        )

        embed_src = self.dropout(
            self.src_word_embedding(src) + self.src_position_embedding(src_positions)
        )
        embed_trg = self.dropout(
            self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask
        )

        out = self.fc_out(out)

        return out
