import torch
import torch.nn as nn
import torch.nn.functional as F

class DrugRiskANN(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_classes=7, num_targets=19):
        super(DrugRiskANN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for _ in range(num_targets)
        ])
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        outputs = [head(x) for head in self.heads]
        
        return torch.stack(outputs, dim=1)
