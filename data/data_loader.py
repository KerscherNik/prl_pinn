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

def get_dataloaders(file_paths, batch_size=32, sequence_length=5, test_size=0.2, verbose=False):
    X_train, X_test, y_train, y_test, scaler = preprocess_data(file_paths, sequence_length, test_size, verbose)

    train_dataset = CartpoleDataset(X_train, y_train)
    test_dataset = CartpoleDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Always output basic loading information
    print(f"Final loaded data:")
    print(f"  Train set: {len(train_dataset)} samples")
    print(f"  Test set: {len(test_dataset)} samples")

    if verbose:
        print("\nDetailed Dataset Information:")
        print(f"Sequence length: {sequence_length}")
        print(f"Batch size: {batch_size}")
        print(f"Test size: {test_size}")
        print(f"\nTrain dataset:")
        print(f"  Shape: {train_dataset.sequences.shape}")
        print(f"  Features min: {torch.min(train_dataset.sequences)}")
        print(f"  Features max: {torch.max(train_dataset.sequences)}")
        print(f"  Targets min: {torch.min(train_dataset.targets)}")
        print(f"  Targets max: {torch.max(train_dataset.targets)}")
        print(f"\nTest dataset:")
        print(f"  Shape: {test_dataset.sequences.shape}")
        print(f"  Features min: {torch.min(test_dataset.sequences)}")
        print(f"  Features max: {torch.max(test_dataset.sequences)}")
        print(f"  Targets min: {torch.min(test_dataset.targets)}")
        print(f"  Targets max: {torch.max(test_dataset.targets)}")
        print(f"\nDataloaders:")
        print(f"  Train batches: {len(train_dataloader)}")
        print(f"  Test batches: {len(test_dataloader)}")

    return train_dataloader, test_dataloader, scaler