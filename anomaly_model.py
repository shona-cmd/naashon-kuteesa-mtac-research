# tiny model for quick prototyping
import torch
import torch.nn as nn

class TinyAnomalyDetector(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

# simple helper to run inference
def predict(model, temp, hum, threshold=0.8):
    model.eval()
    with torch.no_grad():
        t = torch.tensor([[temp, hum]], dtype=torch.float32)
        out = model(t).item()
    # out ranges 0..1; above threshold => anomaly
    return out, out >= threshold
