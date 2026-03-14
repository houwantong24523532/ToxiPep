import pandas as pd
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=51):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbeddingLayer, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.bi_lstm = nn.LSTM(d_model, d_model // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)

    def forward(self, input_ids):
        x = self.src_emb(input_ids)
        embeddings = self.pos_emb(x.transpose(0, 1)).transpose(0, 1)
        lstm_output, _ = self.bi_lstm(embeddings)
        return lstm_output



class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.3):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x



class Structural(nn.Module):
    def __init__(self, embedding_dim=21, max_seq_len=50, filter_num=64, filter_sizes=None):

        super(Structural, self).__init__()
        if filter_sizes is None:
            filter_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)]
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.filter_sizes = filter_sizes
        self.filter_num = filter_num

        self.convs = nn.ModuleList(
            [nn.Conv2d(embedding_dim, filter_num, fsz, stride=1, padding=(fsz[0] // 2, fsz[1] // 2)) for fsz in filter_sizes]
        )

        self.fc = nn.Linear(len(filter_sizes) * filter_num, 1024)
        self.dropout = nn.Dropout(0.3)

    def forward(self, graph, device):

        graph = graph.to(device)
        graph = graph.transpose(2, 3)
        graph = graph.transpose(1, 2)
        conv_outs = [F.relu(conv(graph)) for conv in self.convs]
        pooled_outs = [F.adaptive_avg_pool2d(conv_out, (1, 1)).view(graph.size(0), -1) for conv_out in conv_outs]
        concat_out = torch.cat(pooled_outs, 1)
        representation = self.fc(concat_out)
        representation = self.dropout(representation)
        return representation


class peptide(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_layers, n_heads, max_len=50):
        super(peptide, self).__init__()
        self.emb = EmbeddingLayer(vocab_size, d_model)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        # 降低dropout值，避免模型欠拟合导致无法学习到少数类特征
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, d_model)
        )

    def forward(self, input_ids):
        emb_out = self.emb(input_ids)
        trans_out = self.transformer_blocks(emb_out)
        pooled_output = self.pool(trans_out.transpose(1, 2)).squeeze(-1)
        logits = self.fc(pooled_output)
        return logits


class BilinearAttentionNetwork(nn.Module):
    """BAN: 使用低秩双线性池化 + 多glimpse实现多模态特征融合"""

    def __init__(self, v_dim, q_dim, hidden_dim=128, n_glimpses=4, dropout=0.3):
        super(BilinearAttentionNetwork, self).__init__()
        self.n_glimpses = n_glimpses
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * n_glimpses

        self.v_proj = nn.ModuleList(
            [nn.Linear(v_dim, hidden_dim) for _ in range(n_glimpses)]
        )
        self.q_proj = nn.ModuleList(
            [nn.Linear(q_dim, hidden_dim) for _ in range(n_glimpses)]
        )
        self.bilinear_fc = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_glimpses)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_glimpses)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, q):
        """
        v: (batch, v_dim) — 结构特征
        q: (batch, q_dim) — 肽序列特征
        return: (batch, hidden_dim * n_glimpses)
        """
        glimpses = []
        for i in range(self.n_glimpses):
            v_p = self.v_proj[i](v)
            q_p = self.q_proj[i](q)
            interaction = v_p * q_p  # Hadamard积 — 双线性池化核心
            interaction = self.bilinear_fc[i](interaction)
            interaction = self.norms[i](interaction)
            interaction = F.relu(interaction)
            interaction = self.dropout(interaction)
            glimpses.append(interaction)
        return torch.cat(glimpses, dim=-1)



class ToxiPep_Model(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_layers, n_heads, max_len, structural_config,
                 ban_hidden_dim=128, n_glimpses=4):
        super(ToxiPep_Model, self).__init__()
        self.peptide_model = peptide(vocab_size, d_model, d_ff, n_layers, n_heads, max_len)
        self.structural_model = Structural(**structural_config)
        self.structural_linear = nn.Linear(1024, d_model)

        self.ban = BilinearAttentionNetwork(
            v_dim=d_model, q_dim=d_model,
            hidden_dim=ban_hidden_dim, n_glimpses=n_glimpses
        )

        combined_dim = ban_hidden_dim * n_glimpses
        hidden_dim = 256

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.residual_proj = nn.Linear(combined_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 2)
        self.temperature = 1.0

    def forward(self, input_ids, graph_features, device):
        peptide_output = self.peptide_model(input_ids)
        structural_output = self.structural_model(graph_features, device)
        structural_output = self.structural_linear(structural_output)

        fused = self.ban(structural_output, peptide_output)

        hidden = self.classifier(fused)
        residual = self.residual_proj(fused)
        hidden = hidden + residual

        logits = self.output_layer(hidden) / self.temperature
        return logits
