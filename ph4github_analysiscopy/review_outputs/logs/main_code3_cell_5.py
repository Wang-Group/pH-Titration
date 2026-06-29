import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Fixed random seeds to ensure reproducible results
np.random.seed(42)
torch.manual_seed(42)

# -------------------------------
# global parameters
# -------------------------------
INPUT_DIM = 5    # Features: current pH, target pH, pH change, error (current pH -target pH) and volume dropped in the previous step
HIDDEN_DIM1 = 256
HIDDEN_DIM2 = 256
BATCH_SIZE = 64
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3

# Discrete action space parameters: Volume range [0.01, 10.00] mL, step size 0.01 mL
MIN_VOLUME = 0.01
MAX_VOLUME = 10.0
STEP = 0.01
NUM_ACTIONS = int((MAX_VOLUME - MIN_VOLUME) / STEP) + 1  # 1000 discrete actions

# -------------------------------
# Dataset: Convert continuous labels to discrete categories
# -------------------------------
class VolumePredictionDataset(Dataset):
    def __init__(self, dataset):
        # Convert observations and actions to numpy arrays
        obs = np.array(dataset['Observations'])
        acts = np.array(dataset['Actions'])
        
        # Only keep samples with action category 0 or 2
        mask = np.isin(acts[:, 0], [0, 2])
        obs = obs[mask]
        acts = acts[mask]
        
        # Extract input features: current pH (index 0), target pH (index 1), pH change (index 7), error (current pH -target pH) and the volume dropped in the previous step (index 8)
        current_ph = obs[:, 0]
        target_ph = obs[:, 1]
        ph_change = obs[:, 7]
        error = current_ph - target_ph
        last_added_volume = obs[:, 8]
        inputs = np.stack([current_ph, target_ph, ph_change, error, last_added_volume], axis=1)
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        
        # Extract the volume in action (second column) as the regression target,
        # and converted to discrete categories: category index = round((volume -MIN_VOLUME)/STEP)
        continuous_volumes = acts[:, 1]
        indices = np.rint((continuous_volumes - MIN_VOLUME) / STEP).astype(np.int64)
        self.labels = torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# -------------------------------
# Discrete action strategy model: output NUM_ACTIONS logits, corresponding to discrete volumes of 0.01~10.00 mL
# -------------------------------
class DiscreteVolumeRegressor(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_actions=NUM_ACTIONS):
        super(DiscreteVolumeRegressor, self).__init__()
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM2, num_actions)
        )
        # Generate a list of discrete volumes used to map category indices to actual volumes
        self.discrete_volumes = [round(MIN_VOLUME + i * STEP, 2) for i in range(num_actions)]
    
    def forward(self, x):
        return self.net(x)
    
    # During inference, use argmax to select the category with the highest probability and map it back to the volume
    def predict_volume(self, x):
        logits = self.forward(x)
        _, predicted_indices = torch.max(logits, dim=1)
        # Convert category index to volume
        predicted_volumes = [self.discrete_volumes[idx] for idx in predicted_indices.tolist()]
        return torch.tensor(predicted_volumes, dtype=torch.float32).unsqueeze(1)
    
    # If sampling action is required, Categorical distribution can also be used
    def sample_action(self, x):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)
        volume = self.discrete_volumes[action_index.item()]
        return torch.tensor([[volume]], dtype=torch.float32), log_prob

# -------------------------------
# Tool function: load JSON data
# -------------------------------
def load_json_file(filename):
    with open(filename, 'R') as f:
        data = json.load(f)
    return data

# -------------------------------
# Utility function: Evaluate the model on the data loader (compute cross-entropy loss and accuracy)
# -------------------------------
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    count = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits, dim=1)
            total_correct += (predicted == labels).sum().item()
            count += inputs.size(0)
    avg_loss = total_loss / count
    accuracy = total_correct / count
    return avg_loss, accuracy

# -------------------------------
# Main training process: load train_set_big.json, validation_set_big.json, test_set_big.json for training, verification and testing
# -------------------------------
def main():
    # Load dataset
    train_data = load_json_file('Train set big new1.json')
    val_data = load_json_file('Validation set big new1.json')
    test_data = load_json_file('Test set big new1.json')
    
    train_dataset = VolumePredictionDataset(train_data)
    val_dataset = VolumePredictionDataset(val_data)
    test_dataset = VolumePredictionDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initializing the model, optimizer and loss function (cross-entropy loss)
    model = DiscreteVolumeRegressor(INPUT_DIM, NUM_ACTIONS)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('Inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            logits = model(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_train_loss = total_loss / len(train_dataset)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "Volume regressor best big discrete new1 test.pth")
            print("Saved best model with Val Loss: {:.4f}".format(val_loss))
    
    # testing phase
    model.load_state_dict(torch.load("Volume regressor best big discrete new1 test.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Reasoning example: Example input: [current pH=9.0, target pH=2.0, pH change=-0.5, error=7.0, previous drop volume=0.05]
    example_input = [9.0, 2.0, -0.5, 7.0, 0.05]
    input_tensor = torch.tensor(example_input, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        predicted_volume = model.predict_volume(input_tensor).item()
    print("for input", example_input, "The predicted volume is:", predicted_volume)

if __name__ == 'Main':
    main()
