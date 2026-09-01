import torch
import torch.nn as nn
import numpy as np
import shap
import matplotlib.pyplot as plt
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
def extract_state_vectors(file_path, max_vectors=10):
    state_vectors = []
    state_pattern = r"State = \[\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.+-]+)\s+([\d.+-]+)\]"
    
    with open(file_path, 'R', encoding='Utf 8') as f:
        content = f.read()
    
    # Find all state vectors
    matches = re.finditer(state_pattern, content)
    for match in matches:
        state = [float(match.group(i)) for i in range(1, 6)]
        state_vectors.append(np.array(state, dtype=np.float32))
        if len(state_vectors) >= max_vectors:
            break
    
    return state_vectors[:max_vectors]

# SHAP analysis function
def analyze_shap_importance(state_vectors, model_path="Volume regressor best big discrete new1 trained 1 test.pth", nsamples=500):
    # Initialize model
    model = DiscreteVolumeRegressor()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('Cpu')))
        print("Loading the pre-trained model successfully.")
    except Exception as e:
        print("Failed to load model:", e)
        return None, None
    
    model.eval()
    
    # Wrapping the model into SHAP-available functions
    def model_predict(inputs):
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            outputs = []
            for i in range(inputs.shape[0]):
                volume = model.sample_action(inputs_tensor[i].unsqueeze(0))
                outputs.append(volume)
        return np.array(outputs)
    
    # Convert to NumPy array
    state_vectors_np = np.array(state_vectors, dtype=np.float32)
    
    # Using KernelExplainer
    explainer = shap.KernelExplainer(model_predict, state_vectors_np)
    shap_values = explainer.shap_values(state_vectors_np, nsamples=nsamples)
    
    # Feature name
    feature_names = ['Current ph', 'Target ph', 'P h delta', 'Error', 'Last action volume']
    
    # Calculate average SHAP value
    avg_shap = np.abs(shap_values).mean(axis=0)
    total_shap = avg_shap.sum()
    normalized_shap = avg_shap / total_shap if total_shap > 0 else avg_shap
    
    # Print results
    print("\nAverage shap value (absolute contribution, m l):")
    for name, score in zip(feature_names, avg_shap):
        print(f"{name}: {score:.4f}")
    print("\nNormalized shap value (proportion):")
    for name, score in zip(feature_names, normalized_shap):
        print(f"{name}: {score:.4f}")
    
    # Visualization: Bar Chart
    plt.figure(figsize=(8, 6))
    plt.bar(feature_names, normalized_shap)
    plt.xlabel('feature')
    plt.ylabel('Normalized SHAP value')
    plt.title('SHAP feature importance analysis (first 100 state vectors)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Shap importance.png")
    plt.show()
    
    # Visualization: SHAP Summary Plot
    shap.summary_plot(shap_values, state_vectors_np, feature_names=feature_names, show=False)
    plt.savefig("Shap summary.png")
    plt.show()
    
    return dict(zip(feature_names, avg_shap)), dict(zip(feature_names, normalized_shap))

# main program
if __name__ == "Main":
    # Read txt file and extract up to 100 state vectors
    file_path = "Test output2 modified.txt"
    state_vectors = extract_state_vectors(file_path, max_vectors=500)
    
    print(f"\nExtracted to {len(state_vectors)} a state vector. ")
    
    # Run SHAP analysis
    if state_vectors:
        avg_shap, normalized_shap = analyze_shap_importance(state_vectors)
    else:
        print("The state vector has not been extracted and cannot be analyzed.")