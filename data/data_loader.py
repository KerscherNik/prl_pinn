
import torch
from torch.utils.data import Dataset, DataLoader
from .preprocess import preprocess_data

class CartpoleDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def get_dataloaders(file_paths, batch_size=32, sequence_length=5, test_size=0.2):
    X_train, X_test, y_train, y_test, scaler = preprocess_data(file_paths, sequence_length, test_size)

    train_dataset = CartpoleDataset(X_train, y_train)
    test_dataset = CartpoleDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, scaler