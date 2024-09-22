import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Output to console
                        logging.FileHandler('app.log')  # Log to a file
                    ])

logger = logging.getLogger(__name__)

def load_data(file_paths, verbose=False):
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=';', decimal=',')
        logger.info(f"Loaded {len(df)} samples from {file_path}")
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape}")
    return combined_df

def clean_data(df, verbose=False):
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info("Starting data cleaning process.")
    logger.debug(f"Original data shape: {df.shape}")

    # Handle the datetime column
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    # Convert numeric columns
    numeric_columns = ['cartPos', 'cartVel', 'pendPos', 'pendVel', 'action']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.debug(f"Missing values before cleaning: {df.isnull().sum()}")
    
    # Remove rows where all numeric columns are NaN
    df = df.dropna(subset=numeric_columns, how='all')
    logger.debug(f"After dropping rows with all NaNs - shape: {df.shape}")
    
    # For remaining NaNs, fill with column mean
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())
    
    # Remove extreme outliers (values beyond 5 IQR)
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 5 * IQR
        upper_bound = Q3 + 5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    logger.info(f"Data shape after cleaning: {df.shape}")
    return df

def normalize_data(df, verbose=False):
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    scaler = StandardScaler()
    logger.info("Starting data normalization.")
    
    # Separate the 'action' column from the rest of the data
    action = df['action']
    features = df.drop(columns=['action'])
    
    # Normalize the features
    normalized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    
    # Combine the normalized features with the 'action' column
    normalized_df = pd.concat([normalized_features, action.reset_index(drop=True)], axis=1)
    logger.debug(f"Normalized data shape: {normalized_df.shape}")
    
    return normalized_df, scaler

def create_sequences(df, sequence_length, verbose=False):
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info(f"Creating sequences of length {sequence_length}.")
    sequences = []
    targets = []
    
    for i in range(len(df) - sequence_length):
        seq = df.iloc[i:i+sequence_length].values
        target = df.iloc[i+sequence_length].values
        sequences.append(seq)
        targets.append(target)
    
    logger.info(f"Created {len(sequences)} sequences.")
    return np.array(sequences), np.array(targets)

def split_data(sequences, targets, test_size=0.2, random_state=42, verbose=False):
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info(f"Splitting data into train and test sets with test size {test_size}.")
    X_train, X_test, y_train, y_test = train_test_split(sequences, targets, test_size=test_size, random_state=random_state)
    logger.debug(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def preprocess_data(file_paths, sequence_length=5, test_size=0.2, verbose=False):
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.info("Starting data preprocessing.")
    
    # Load data
    df = load_data(file_paths, verbose)
    
    # Clean data
    df = clean_data(df, verbose)
    
    if df.empty:
        logger.error("All data was filtered out during cleaning.")
        raise ValueError("All data was filtered out during cleaning. Please check your data and cleaning criteria.")
    
    # Normalize data
    normalized_df, scaler = normalize_data(df[['cartPos', 'cartVel', 'pendPos', 'pendVel', 'action']], verbose)
    
    # Create sequences
    sequences, targets = create_sequences(normalized_df, sequence_length, verbose)
    
    if len(sequences) == 0:
        logger.error("No sequences could be created. Check the sequence_length parameter.")
        raise ValueError("No sequences could be created. Please check your sequence_length parameter.")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(sequences, targets, test_size=test_size, verbose=verbose)
    
    logger.info("Data preprocessing complete.")
    return X_train, X_test, y_train, y_test, scaler
