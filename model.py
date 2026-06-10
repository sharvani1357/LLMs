import torch
import torch.nn as nn

# ==========================================
# FEED FORWARD NETWORK
# ==========================================

class FeedForwardNetwork(nn.Module):

    def __init__(
            self,
            embed_dim,
            ff_dim):

        super().__init__()

        self.fc1 = nn.Linear(
            embed_dim,
            ff_dim
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            ff_dim,
            embed_dim
        )

    def forward(self, x):

        return self.fc2(
            self.relu(
                self.fc1(x)
            )
        )

# ==========================================
# GPT DECODER BLOCK
# ==========================================

class GPTDecoderBlock(nn.Module):

    def __init__(
            self,
            embed_dim,
            num_heads,
            ff_dim,
            dropout):

        super().__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.ffn = FeedForwardNetwork(
            embed_dim,
            ff_dim
        )

        self.dropout2 = nn.Dropout(
            dropout
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

    def forward(self, x):

        seq_len = x.size(1)

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len
            ),
            diagonal=1
        ).bool().to(
            x.device
        )

        attention_output, _ = self.mha(
            x,
            x,
            x,
            attn_mask=mask
        )

        attention_output = self.dropout1(
            attention_output
        )

        x = self.norm1(
            x + attention_output
        )

        ffn_output = self.ffn(x)

        ffn_output = self.dropout2(
            ffn_output
        )

        x = self.norm2(
            x + ffn_output
        )

        return x

# ==========================================
# MINI GPT MODEL
# ==========================================

class MiniGPT(nn.Module):

    def __init__(
            self,
            vocab_size,
            embed_dim,
            max_seq_len,
            num_heads,
            num_layers,
            ff_dim,
            dropout):

        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            embed_dim
        )

        self.decoder_blocks = nn.ModuleList(

            [

                GPTDecoderBlock(
                    embed_dim,
                    num_heads,
                    ff_dim,
                    dropout
                )

                for _ in range(
                    num_layers
                )

            ]

        )

        self.fc_out = nn.Linear(
            embed_dim,
            vocab_size
        )

    def forward(self, x):

        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        ).unsqueeze(0)

        x = (

            self.token_embedding(x)

            +

            self.position_embedding(
                positions
            )

        )

        for block in self.decoder_blocks:

            x = block(x)

        logits = self.fc_out(x)

        return logits