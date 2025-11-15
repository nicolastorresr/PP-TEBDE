import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .dataset import PPTEBDEDataset

def load_dataset(dataset_name, config):
    """
    Load and preprocess the specified dataset.
    """
    if dataset_name == 'edurec':
        return load_edurec_dataset(config)
    elif dataset_name == 'movielens':
        return load_movielens_dataset(config)
    elif dataset_name == 'amazon':
        return load_amazon_dataset(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_edurec_dataset(config):
    """
    Load and preprocess the EduRec-2024 dataset.
    """
    # Load raw data
    df = pd.read_csv(config.edurec_path, 
                     names=['UserID', 'CourseID', 'Section', 'Timestamp', 'Rating', 'SCE'])
    
    # Preprocess
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).astype(int) / 10**9  # Convert to Unix timestamp
    df['UserID'] = df['UserID'].astype('category').cat.codes
    df['CourseID'] = df['CourseID'].astype('category').cat.codes
    
    # Convert ratings to binary feedback (>=55 is positive)
    df['Rating'] = (df['Rating'] >= 55).astype(int)
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'UserID': 'user_id',
        'CourseID': 'item_id',
        'Timestamp': 'timestamp',
        'Rating': 'rating'
    })
    
    # Select relevant columns (excluding SCE)
    df = df[['user_id', 'item_id', 'timestamp', 'rating']]
    
    return create_train_val_test_split(df, config)

def load_movielens_dataset(config):
    """
    Load and preprocess the MovieLens dataset.
    """
    # Load raw data
    df = pd.read_csv(config.movielens_path)
    
    # Preprocess
    df['user_id'] = df['userId'].astype('category').cat.codes
    df['item_id'] = df['movieId'].astype('category').cat.codes
    df['timestamp'] = df['timestamp'].astype(int)
    df['rating'] = (df['rating'] > 3.5).astype(int)  # Convert ratings to binary feedback
    
    return create_train_val_test_split(df, config)

def load_amazon_dataset(config):
    """
    Load and preprocess the Amazon Electronics dataset.
    """
    # Load raw data
    df = pd.read_json(config.amazon_path, lines=True)
    
    # Preprocess
    df['user_id'] = df['reviewerID'].astype('category').cat.codes
    df['item_id'] = df['asin'].astype('category').cat.codes
    df['timestamp'] = pd.to_datetime(df['unixReviewTime'], unit='s').astype(int) / 10**9
    df['rating'] = (df['overall'] > 3).astype(int)  # Convert ratings to binary feedback
    
    return create_train_val_test_split(df, config)

def create_train_val_test_split(df, config):
    """
    Create train, validation, and test splits maintaining temporal order.
    """
    # Sort by timestamp GLOBALLY
    df = df.sort_values('timestamp')
    
    # Calculate split points based on time, not rows
    total_time = df['timestamp'].max() - df['timestamp'].min()
    train_cutoff = df['timestamp'].min() + total_time * (1 - config.test_size - config.val_size)
    val_cutoff = df['timestamp'].min() + total_time * (1 - config.test_size)
    
    # Split based on temporal cutoffs
    train = df[df['timestamp'] < train_cutoff]
    val = df[(df['timestamp'] >= train_cutoff) & (df['timestamp'] < val_cutoff)]
    test = df[df['timestamp'] >= val_cutoff]
    
    # Create DataLoader objects
    train_dataset = PPTEBDEDataset(train)
    val_dataset = PPTEBDEDataset(val)
    test_dataset = PPTEBDEDataset(test)
    
    # CRITICAL: shuffle=False for temporal data!
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
