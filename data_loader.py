import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CartpoleDataset(Dataset):
    def __init__(self, dataframe):
        # Keep only the numerical columns
        self.data = dataframe[['cartPos', 'cartVel', 'pendPos', 'pendVel', 'action']]
        self.data = self.data.astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return torch.tensor(row.values, dtype=torch.float32)

def load_and_combine_data(file_paths):
    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=';', decimal=',')  # Use semicolon as separator and comma as decimal
        print(f"Loaded {len(df)} samples from {file_path}")
        #print(df.head())
        #print("Columns: ", df.columns)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def split_data(dataframe, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(dataframe, test_size=test_size, random_state=random_state)
    return train_df, test_df

def get_dataloaders(file_paths, batch_size=32, test_size=0.2):
    combined_df = load_and_combine_data(file_paths)
    train_df, test_df = split_data(combined_df, test_size=test_size)

    train_dataset = CartpoleDataset(train_df)
    test_dataset = CartpoleDataset(test_df)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader