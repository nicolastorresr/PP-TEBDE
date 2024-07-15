import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, dropout_rate=0.1):
        super(TemporalAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        assert self.head_dim * num_heads == embedding_dim, "Embedding dimension must be divisible by number of heads"
        
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Project inputs
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(query.device)
        
        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # Compute attention weights
        attention = F.softmax(energy, dim=-1)
        
        # Apply dropout
        attention = self.dropout(attention)
        
        # Compute output
        x = torch.matmul(attention, V)
        
        # Reshape and project output
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.embedding_dim)
        x = self.fc_out(x)
        
        return x, attention

class TemporalAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, dropout_rate=0.1):
        super(TemporalAttentionLayer, self).__init__()
        
        self.attention = TemporalAttention(embedding_dim, num_heads, dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

class TemporalAttentionModule(nn.Module):
    def __init__(self, embedding_dim, num_layers=2, num_heads=4, dropout_rate=0.1):
        super(TemporalAttentionModule, self).__init__()
        
        self.layers = nn.ModuleList([
            TemporalAttentionLayer(embedding_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        attention_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        return x, attention_weights

    def get_attention_weights(self, x, mask=None):
        with torch.no_grad():
            _, attention_weights = self.forward(x, mask)
        return attention_weights
