import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_paths):
    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=';', decimal=',')
        print(f"Loaded {len(df)} samples from {file_path}")
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def clean_data(df):
    print("Original data shape:", df.shape)
    print("Columns:", df.columns)
    print("Data types:", df.dtypes)
    
    # Handle the datetime column
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    # Convert numeric columns, keeping original values if conversion fails
    numeric_columns = ['cartPos', 'cartVel', 'pendPos', 'pendVel', 'action']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("After conversion - shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    
    # Remove rows where all numeric columns are NaN
    df = df.dropna(subset=numeric_columns, how='all')
    print("After dropping rows with all NaNs in numeric columns - shape:", df.shape)
    
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
    
    print("After removing extreme outliers - shape:", df.shape)
    
    return df

def normalize_data(df):
    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return normalized_df, scaler

def create_sequences(df, sequence_length):
    sequences = []
    targets = []
    
    for i in range(len(df) - sequence_length):
        seq = df.iloc[i:i+sequence_length].values
        target = df.iloc[i+sequence_length].values
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def split_data(sequences, targets, test_size=0.2, random_state=42):
    return train_test_split(sequences, targets, test_size=test_size, random_state=random_state)

def preprocess_data(file_paths, sequence_length=5, test_size=0.2):
    # Load data
    df = load_data(file_paths)
    print(f"Loaded data shape: {df.shape}")
    
    # Clean data
    df = clean_data(df)
    print(f"Cleaned data shape: {df.shape}")
    
    if df.empty:
        raise ValueError("All data was filtered out during cleaning. Please check your data and cleaning criteria.")
    
    # Normalize data
    normalized_df, scaler = normalize_data(df[['cartPos', 'cartVel', 'pendPos', 'pendVel', 'action']])
    print(f"Normalized data shape: {normalized_df.shape}")
    
    # Create sequences
    sequences, targets = create_sequences(normalized_df, sequence_length)
    print(f"Sequences shape: {sequences.shape}, Targets shape: {targets.shape}")
    
    if len(sequences) == 0:
        raise ValueError("No sequences could be created. Please check your sequence_length parameter.")
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(sequences, targets, test_size=test_size)
    
    print(f"Train shapes - X: {X_train.shape}, y: {y_train.shape}")
    print(f"Test shapes - X: {X_test.shape}, y: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

# Example usage
if __name__ == "__main__":
    file_paths = ["data/cartpole_data.csv"]
    X_train, X_test, y_train, y_test, scaler = preprocess_data(file_paths)
    print("Preprocessed data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")