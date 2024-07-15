import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_time_bins, dropout_rate=0.2):
        super(TemporalCF, self).__init__()
        
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.time_embeddings = nn.Embedding(num_time_bins, embedding_dim)
        
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        self.time_decay = nn.Parameter(torch.FloatTensor([0.1]))
        
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, 1)
        )
        
        self.num_time_bins = num_time_bins
        self.init_weights()

    def init_weights(self):
        for embedding in [self.user_embeddings, self.item_embeddings, self.time_embeddings]:
            nn.init.normal_(embedding.weight, std=0.01)
        
        for bias in [self.user_bias, self.item_bias]:
            nn.init.zeros_(bias.weight)
        
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_ids, item_ids, timestamps):
        user_embed = self.user_embeddings(user_ids)
        item_embed = self.item_embeddings(item_ids)
        
        # Convert timestamps to time bin indices
        time_bin_indices = self.timestamp_to_bin(timestamps)
        time_embed = self.time_embeddings(time_bin_indices)
        
        # Compute time decay factor
        time_diff = timestamps.float() - timestamps.float().min()
        time_decay_factor = torch.exp(-self.time_decay * time_diff).unsqueeze(1)
        
        # Apply time decay to user and item embeddings
        user_embed = user_embed * time_decay_factor
        item_embed = item_embed * time_decay_factor
        
        # Concatenate embeddings
        concat_embed = torch.cat([user_embed, item_embed, time_embed], dim=1)
        
        # Pass through fully connected layers
        output = self.fc_layers(concat_embed)
        
        # Add user and item biases
        output += self.user_bias(user_ids) + self.item_bias(item_ids)
        
        return output.squeeze()

    def timestamp_to_bin(self, timestamps):
        # Convert timestamps to bin indices
        # This is a simple linear binning strategy; you might want to use a more sophisticated approach
        min_timestamp = timestamps.min()
        max_timestamp = timestamps.max()
        bin_size = (max_timestamp - min_timestamp) / self.num_time_bins
        bin_indices = ((timestamps - min_timestamp) / bin_size).long().clamp(0, self.num_time_bins - 1)
        return bin_indices

    def get_temporal_user_embedding(self, user_id, timestamp):
        user_embed = self.user_embeddings(user_id)
        time_bin = self.timestamp_to_bin(timestamp)
        time_embed = self.time_embeddings(time_bin)
        time_diff = timestamp.float() - timestamp.float().min()
        time_decay_factor = torch.exp(-self.time_decay * time_diff).unsqueeze(1)
        return user_embed * time_decay_factor + time_embed

    def get_temporal_item_embedding(self, item_id, timestamp):
        item_embed = self.item_embeddings(item_id)
        time_bin = self.timestamp_to_bin(timestamp)
        time_embed = self.time_embeddings(time_bin)
        time_diff = timestamp.float() - timestamp.float().min()
        time_decay_factor = torch.exp(-self.time_decay * time_decay_factor).unsqueeze(1)
        return item_embed * time_decay_factor + time_embed
