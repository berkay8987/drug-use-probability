import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from model import DrugRiskANN
import os

def train_model(data_path='drug_consumption.data', num_epochs=100, batch_size=32, learning_rate=0.001):
    # cuda = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, num_targets = load_data(data_path, batch_size=batch_size)
    
    model = DrugRiskANN(num_targets=num_targets).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            
            loss = 0
            for i in range(num_targets):
                loss += criterion(outputs[:, i, :], targets[:, i])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                outputs = model(features)
                
                loss = 0
                for i in range(num_targets):
                    loss += criterion(outputs[:, i, :], targets[:, i])
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 2)
                correct_predictions += (predicted == targets).sum().item()
                total_predictions += targets.numel()
        
        val_loss /= len(test_loader)
        accuracy = correct_predictions / total_predictions
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
    print("Training finished!")
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    target_cols = [
        'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 
        'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 
        'Mushrooms', 'Nicotine', 'Semer', 'VSA'
    ]
    
    correct_per_drug = {drug: 0 for drug in target_cols}
    total_per_drug = {drug: 0 for drug in target_cols}
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs, 2)
            
            for i, drug in enumerate(target_cols):
                correct_per_drug[drug] += (predicted[:, i] == targets[:, i]).sum().item()
                total_per_drug[drug] += targets.size(0)
                
    print("\nAccuracy per drug:")
    for drug in target_cols:
        acc = correct_per_drug[drug] / total_per_drug[drug]
        print(f"{drug}: {acc:.4f}")

if __name__ == "__main__":
    if os.path.exists('drug_consumption.data'):
        trained_model = train_model()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, test_loader, _ = load_data('drug_consumption.data')
        evaluate_model(trained_model, test_loader, device)
    else:
        print("Data file not found!")
