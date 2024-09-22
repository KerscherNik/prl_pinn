import torch
from torch.utils.data import Dataset, DataLoader
from .preprocess import preprocess_data
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Output to console
                        logging.FileHandler('app.log')  # Log to a file
                    ])

logger = logging.getLogger(__name__)

class CartpoleDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        logger.info(f"Initialized CartpoleDataset with {len(sequences)} samples.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def get_dataloaders(file_paths, batch_size=32, sequence_length=5, test_size=0.2, verbose=False):
    logger.info("Starting dataloader creation.")
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(file_paths, sequence_length, test_size, verbose)
    
    # Create datasets
    train_dataset = CartpoleDataset(X_train, y_train)
    test_dataset = CartpoleDataset(X_test, y_test)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Data loaded successfully. Train set: {len(train_dataset)}, Test set: {len(test_dataset)}")
    
    if verbose:
        logger.debug(f"Detailed Dataset Information: "
                     f"Sequence length: {sequence_length}, Batch size: {batch_size}, Test size: {test_size}")
        logger.debug(f"Train dataset shape: {train_dataset.sequences.shape}, Test dataset shape: {test_dataset.sequences.shape}")
    
    return train_dataloader, test_dataloader, scaler
