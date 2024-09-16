import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Custom CartpoleDataset: Ensures the relevant columns,
# enables loading data from multiple data sources at once,
# split data in train and test data
class CartpoleDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe[['cartPos', 'cartVel', 'pendPos', 'pendVel', 'action']].astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return (
            torch.tensor(row['cartPos'], dtype=torch.float32),
            torch.tensor(row['cartVel'], dtype=torch.float32),
            torch.tensor(row['pendPos'], dtype=torch.float32),
            torch.tensor(row['pendVel'], dtype=torch.float32),
            torch.tensor(row['action'], dtype=torch.float32)
        )

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

    # Debug: Print the shape of the first batch
    for batch in train_dataloader:
        print("Batch shape:", [b.shape for b in batch])
        print("Number of elements in batch:", len(batch))
        break

    return train_dataloader, test_dataloader