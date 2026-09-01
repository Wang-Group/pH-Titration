# Source notebook: main_code3.ipynb
# Raw notebook cell index: 30
# Code-cell export index: 30
# First non-empty line: import torch
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import re

# fixed random seed
seed = 555
torch.manual_seed(seed)
np.random.seed(seed)

# discrete action strategy model
class DiscreteVolumeRegressor(nn.Module):
    def __init__(self, input_dim=5, min_volume=0.01, max_volume=10.0, step=0.01):
        super(DiscreteVolumeRegressor, self).__init__()
        self.discrete_volumes = [round(min_volume + i * step, 2) for i in range(int((max_volume - min_volume) / step) + 1)]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.discrete_volumes))
        )
    
    def forward(self, x):
        return self.net(x)
    
    def sample_action(self, x):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action_index = dist.sample()
        volume = self.discrete_volumes[action_index.item()]
        return volume

# Extract state vectors from txt file, up to 100
def extract_state_vectors(file_path, max_vectors=10000):
    state_vectors = []
    state_pattern = r"State = \[\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.+-]+)\s+([\d.+-]+)\]"
    
    with open(file_path, 'R', encoding='Utf 8') as f:
        content = f.read()
    
    matches = re.finditer(state_pattern, content)
    for match in matches:
        state = [float(match.group(i)) for i in range(1, 6)]
        state_vectors.append(np.array(state, dtype=np.float32))
        if len(state_vectors) >= max_vectors:
            break
    
    return state_vectors[:max_vectors]

# Correlation analysis function
def correlation_analysis(file_path, model_path="Volume regressor best big discrete new1 trained 1 test.pth", max_vectors=20000):
    # Extract state vector
    state_vectors = extract_state_vectors(file_path, max_vectors)
    if not state_vectors:
        print("The state vector has not been extracted and cannot be analyzed.")
        return None, None
    
    print(f"\nExtracted to {len(state_vectors)} a state vector. ")
    
    # Initialize model
    model = DiscreteVolumeRegressor()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('Cpu')))
        print("Loading the pre-trained model successfully.")
    except Exception as e:
        print("Failed to load model:", e)
        return None, None
    
    model.eval()
    
    # Get predicted volume
    state_vectors_np = np.array(state_vectors, dtype=np.float32)
    volumes = []
    with torch.no_grad():
        for state in state_vectors_np:
            volume = model.sample_action(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            volumes.append(volume)
    volumes = np.array(volumes)
    
    # Feature name
    feature_names = ['Current ph', 'Target ph', 'P h delta', 'Error', 'Last action volume']
    
    # Calculate correlation coefficient
    pearson_corrs = {}
    spearman_corrs = {}
    for i, name in enumerate(feature_names):
        # Take the absolute value of error
        if name == 'Error':
            feature_values = np.abs(state_vectors_np[:, i])
        else:
            feature_values = state_vectors_np[:, i]
        pearson_corr, pearson_p = pearsonr(feature_values, volumes)
        spearman_corr, spearman_p = spearmanr(feature_values, volumes)
        pearson_corrs[name] = (pearson_corr, pearson_p)
        spearman_corrs[name] = (spearman_corr, spearman_p)
    
    # Print results
    print("\nPearson correlation coefficient (correlation coefficient, p-value):")
    for name, (corr, p) in pearson_corrs.items():
        print(f"{name}: {corr:.4f} (p={p:.4f})")
    print("\nSpearman correlation coefficient (correlation coefficient, p-value):")
    for name, (corr, p) in spearman_corrs.items():
        print(f"{name}: {corr:.4f} (p={p:.4f})")
    
    # Visualization: Bar Chart
    plt.figure(figsize=(10, 6))
    corr_df = pd.DataFrame({
        'Pearson': [corr for corr, _ in pearson_corrs.values()],
        'Spearman': [corr for corr, _ in spearman_corrs.values()]
    }, index=feature_names)
    corr_df.plot(kind='Bar', ax=plt.gca())
    plt.xlabel('feature')
    plt.ylabel('Correlation coefficient')
    plt.title('Correlation analysis between state vectors and predicted volumes (first 10,000 state vectors)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Correlation bar.png")
    plt.show()
    
    # Visualization: Heatmap
    plt.figure(figsize=(8, 6))
    corr_matrix = np.zeros((len(feature_names), 2))
    for i, name in enumerate(feature_names):
        corr_matrix[i, 0] = pearson_corrs[name][0]
        corr_matrix[i, 1] = spearman_corrs[name][0]
    sns.heatmap(corr_matrix, annot=True, xticklabels=['Pearson', 'Spearman'], yticklabels=feature_names, cmap='Coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation heat map')
    plt.tight_layout()
    plt.savefig("Correlation heatmap.png")
    plt.show()
    
    return pearson_corrs, spearman_corrs

# main program
if __name__ == "Main":
    file_path = "Test output2 modified.txt"
    pearson_corrs, spearman_corrs = correlation_analysis(file_path)