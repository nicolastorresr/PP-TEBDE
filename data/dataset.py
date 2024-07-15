import torch
from torch.utils.data import Dataset

class PPTEBDEDataset(Dataset):
    def __init__(self, df):
        self.users = torch.LongTensor(df['user_id'].values)
        self.items = torch.LongTensor(df['item_id'].values)
        self.timestamps = torch.FloatTensor(df['timestamp'].values)
        self.labels = torch.FloatTensor(df['rating'].values)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.timestamps[idx], self.labels[idx]
