import torch


import torch.nn as nn

class AnomalyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


if __name__ == '__main__':
    # Example usage
    model = AnomalyDetector()
    print(model)

    # Generate dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,)).float()

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 10
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred.squeeze(), y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch+1}: Loss = {loss.item():.4f}')

    # Evaluate the model
    with torch.no_grad():
        y_pred = model(X)
        y_pred_class = (y_pred > 0.5).float()
        accuracy = (y_pred_class.squeeze() == y).sum() / float(y.shape[0])
        print(f'Accuracy: {accuracy:.4f}')
