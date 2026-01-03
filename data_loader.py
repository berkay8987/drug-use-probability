import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DrugDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features.values)
        self.targets = torch.LongTensor(targets.values)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def load_data(file_path, batch_size=32, test_size=0.2, random_state=42):
    feature_cols = [
        'ID', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 
        'Impulsive', 'SS', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity'
    ]
    target_cols = [
        'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 
        'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 
        'Mushrooms', 'Nicotine', 'Semer', 'VSA'
    ]
    col_names = feature_cols + target_cols
    
    df = pd.read_csv(file_path, names=col_names)
    
    df = df.drop('ID', axis=1)
    
    X = df.iloc[:, :12]
    y = df.iloc[:, 12:]

    label_mapping = {
        'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3, 
        'CL4': 4, 'CL5': 5, 'CL6': 6
    }
    
    for col in y.columns:
        y[col] = y[col].map(label_mapping)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    train_dataset = DrugDataset(X_train, y_train)
    test_dataset = DrugDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, len(target_cols)

if __name__ == "__main__":
    train_loader, test_loader, num_targets = load_data('drug_consumption.data')
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Number of targets: {num_targets}")
    
    features, targets = next(iter(train_loader))
    print(f"Feature shape: {features.shape}")
    print(f"Target shape: {targets.shape}")
