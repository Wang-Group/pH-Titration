# Source notebook: main_code3.ipynb
# Raw notebook cell index: 8
# Code-cell export index: 8
# First non-empty line: import torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

# -------------------------------
# Data set class, consistent with training
# -------------------------------
class VolumePredictionDataset(Dataset):
    def __init__(self, dataset):
        # Convert to numpy array
        obs = np.array(dataset['Observations'])
        acts = np.array(dataset['Actions'])
        
        # Only keep samples with action categories 0 and 2
        mask = np.isin(acts[:, 0], [0, 2])
        obs = obs[mask]
        acts = acts[mask]
        
        # Extract input features:
        # Current pH (index 0), target pH (index 1), pH change (index 7), error (current pH -target pH)
        # and the volume added in the previous step (index 8)
        current_ph = obs[:, 0]
        target_ph = obs[:, 1]
        ph_change = obs[:, 7]
        error = current_ph - target_ph
        last_added_volume = obs[:, 8]
        inputs = np.stack([current_ph, target_ph, ph_change, error, last_added_volume], axis=1)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        
        # Label: Volume in action (second value), as regression target
        self.labels = torch.tensor(acts[:, 1], dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# -------------------------------
# discrete action strategy model
# -------------------------------
INPUT_DIM = 5
HIDDEN_DIM1 = 256
HIDDEN_DIM2 = 256

class DiscreteVolumeRegressor(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, min_volume=0.01, max_volume=10.0, step=0.01):
        super(DiscreteVolumeRegressor, self).__init__()
        # Generate discrete action list: 0.01 ~ 10.00 mL
        self.discrete_volumes = [round(min_volume + i * step, 2)
                                 for i in range(int((max_volume - min_volume) / step) + 1)]
        self.num_actions = len(self.discrete_volumes)
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM2, self.num_actions)
        )
    
    def forward(self, x):
        return self.net(x)
    
    # During inference, use argmax to select the category with the highest probability, and then map it to the actual volume
    def predict_volume(self, x):
        logits = self.forward(x)
        _, predicted_indices = torch.max(logits, dim=1)
        predicted_volume = self.discrete_volumes[predicted_indices.item()]
        return torch.tensor([[predicted_volume]], dtype=torch.float32)

# -------------------------------
# Tool function: load JSON data
# -------------------------------
def load_json_file(filename):
    with open(filename, 'R') as f:
        data = json.load(f)
    return data

# -------------------------------
# Test set evaluation function: Calculate MSE, MAE and R²
# -------------------------------
def evaluate_model(model, dataloader):
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    
    total_mse_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            # For each sample in each batch, use predict_volume to get the predicted volume
            batch_preds = []
            for i in range(inputs.size(0)):
                x = inputs[i].unsqueeze(0)
                pred = model.predict_volume(x)
                batch_preds.append(pred)
            batch_preds = torch.cat(batch_preds, dim=0)
            
            mse_loss = mse_criterion(batch_preds, labels)
            total_mse_loss += mse_loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            all_preds.append(batch_preds)
            all_labels.append(labels)
    
    avg_mse_loss = total_mse_loss / total_samples
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    mae_loss = mae_criterion(all_preds, all_labels).item()
    
    ss_res = torch.sum((all_labels - all_preds) ** 2)
    mean_labels = torch.mean(all_labels)
    ss_tot = torch.sum((all_labels - mean_labels) ** 2)
    r2_score = 1 - ss_res / ss_tot
    
    print("Test MSE Loss: {:.4f}".format(avg_mse_loss))
    print("Test MAE Loss: {:.4f}".format(mae_loss))
    print("Test R² Score: {:.4f}".format(r2_score.item()))

# -------------------------------
# Main test process
# -------------------------------
if __name__ == 'Main':
    # Load test set data
    test_data = load_json_file('Test set big new1.json')
    test_dataset = VolumePredictionDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize the discrete model and load the pre-trained model state
    model = DiscreteVolumeRegressor(INPUT_DIM, min_volume=0.01, max_volume=10.0, step=0.01)
    model.load_state_dict(torch.load("Volume regressor best big discrete new1 test.pth", map_location=torch.device('Cpu')))
    model.eval()
    
    # Evaluate the model on the test set
    evaluate_model(model, test_loader)
