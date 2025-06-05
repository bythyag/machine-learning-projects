import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import datetime
import time
import os # Import the os module

# Synthetic Data Generation
def generate_ev_battery_data(num_samples=2000, cycles=5):
    """Generates synthetic EV battery data."""
    time_index = np.arange(num_samples)

    # Age: Linearly increasing (e.g., in days or cycles)
    age = np.linspace(0, 500, num_samples) # Simulating ~1.5 years of use

    # SOH (State of Health): Starts near 100% and degrades slowly, non-linearly with age
    soh_degradation = 0.05 * (age / age.max())**1.5 + np.random.normal(0, 0.005, num_samples)
    soh = 1.0 - np.clip(soh_degradation, 0, 0.2) # Cap degradation at 20%

    # SOC (State of Charge): Simulate charge/discharge cycles using sine wave
    soc_base = 0.5 + 0.4 * np.sin(2 * np.pi * cycles * time_index / num_samples)
    soc_noise = np.random.normal(0, 0.02, num_samples)
    soc_drift = -0.05 * (age / age.max()) # Slight decrease in max SOC capacity with age
    soc = np.clip(soc_base + soc_noise + soc_drift, 0.05, 0.95) # Keep SOC within reasonable bounds (5% to 95%)

    # Current: Correlated with SOC changes (derivative) + noise + some spikes
    soc_diff = np.gradient(soc)
    current_base = -soc_diff * 500 # Scaling factor for current magnitude
    current_noise = np.random.normal(0, 1.5, num_samples) # Increased noise a bit
    # Add some random high current spikes (charge/discharge events)
    num_spikes = 40
    spike_indices = np.random.randint(0, num_samples, num_spikes)
    spike_values = np.random.uniform(-60, 60, num_spikes) # Amps
    current_spikes = np.zeros(num_samples)
    current_spikes[spike_indices] = spike_values
    current = current_base + current_noise + current_spikes + np.random.normal(0, 0.8) # Base noise

    # Voltage: Primarily correlated with SOC, but affected by current (Ohmic drop/rise) and temp
    voltage_soc_effect = 3.0 + 1.2 * soc # Nominal voltage range (e.g., 3.0V to 4.2V)
    voltage_current_effect = -current * 0.005 # Internal resistance effect (adjust multiplier as needed)
    voltage_soh_effect = -0.1 * (1 - soh) # Lower voltage slightly with lower SOH
    voltage_noise = np.random.normal(0, 0.015, num_samples)
    voltage = voltage_soc_effect + voltage_current_effect + voltage_soh_effect + voltage_noise

    # Temperature: Increases with high absolute current, slight drift with age/SOH, + noise
    temp_current_effect = np.abs(current) * 0.08 # Temperature rise due to current (adjusted factor)
    temp_base = 25.0 + 5 * (age / age.max()) - 3 * (1-soh) # Baseline temp increases with age, decreases slightly if SOH drops faster
    temp_noise = np.random.normal(0, 0.5, num_samples)
    temp = temp_base + temp_current_effect + temp_noise
    temp = np.clip(temp, 15, 55) # Keep temperature in a reasonable range (15C to 55C)

    # Create Timestamp index
    start_date = datetime.datetime(2023, 1, 1)
    timestamps = [start_date + datetime.timedelta(hours=i) for i in range(num_samples)] # Assume hourly data

    # Create DataFrame
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Voltage': voltage,
        'Current': current,
        'Temperature': temp,
        'Age': age,
        'SOC': soc,
        'SOH': soh
    })
    df = df.set_index('Timestamp')
    return df

# 2. Data Preprocessing
def create_sequences(data, seq_length):
    """Creates sequences and corresponding labels for LSTM."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class TimeSeriesDataset(Dataset):
    """Custom PyTorch Dataset for time series data."""
    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer: batch_first=True expects input shape (batch, seq_len, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)

        # Optional: Dropout layer after LSTM but before Linear
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected layer to map LSTM output to desired output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        # Shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass input through LSTM
        # out contains output features for each time step
        # hn, cn are the final hidden and cell states
        out, _ = self.lstm(x, (h0, c0))

        # We only need the output from the last time step for prediction
        # out shape: (batch_size, seq_len, hidden_size) -> Select last time step
        out = out[:, -1, :] # shape: (batch_size, hidden_size)

        # Apply dropout
        out = self.dropout(out)

        # Pass through the final fully connected layer
        out = self.fc(out) # shape: (batch_size, output_size)
        return out

# Main Execution
if __name__ == "__main__":
    # --- Configuration ---
    N_SAMPLES = 3000       # Number of data points to generate
    N_CYCLES = 8           # Number of charge/discharge cycles in the generated data
    SEQ_LENGTH = 60        # Use last 60 data points (e.g., hours) to predict the next one
    TRAIN_SPLIT = 0.8      # 80% for training, 20% for testing/validation
    BATCH_SIZE = 64        # Number of sequences per training batch
    HIDDEN_SIZE = 100      # Number of neurons in LSTM hidden layer
    NUM_LSTM_LAYERS = 2    # Number of stacked LSTM layers
    DROPOUT_PROB = 0.2     # Dropout probability
    LEARNING_RATE = 0.001  # Optimizer learning rate
    EPOCHS = 30           # Number of training epochs
    N_FUTURE_STEPS = 100   # How many steps into the future to predict
    OUTPUT_DIR = "/Users/thyag/Desktop/projects/eloctrocute/sample_data/lstm_output" # Directory to save images

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Select device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Generate Data
    print("Generating synthetic EV battery data...")
    data_df = generate_ev_battery_data(num_samples=N_SAMPLES, cycles=N_CYCLES)
    print("Data generated successfully.")
    print(data_df.head())
    print(f"\nData shape: {data_df.shape}")

    # Select features to use
    features = ['Voltage', 'Current', 'Temperature', 'Age', 'SOC', 'SOH']
    data = data_df[features].values # Convert to numpy array

    # 2. Preprocess Data 
    print("\nPreprocessing data...")
    # Scale features to be between 0 and 1
    scaler = MinMaxScaler()
    # Fit scaler only on training data to prevent leakage
    split_idx = int(len(data) * TRAIN_SPLIT)
    scaler.fit(data[:split_idx])
    scaled_data = scaler.transform(data)

    # Create sequences
    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    # Split into training and testing sets (chronological split for time series)
    X_train, X_test = X[:split_idx - SEQ_LENGTH], X[split_idx - SEQ_LENGTH:]
    y_train, y_test = y[:split_idx - SEQ_LENGTH], y[split_idx - SEQ_LENGTH:]

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Create PyTorch Datasets and DataLoaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # No shuffle for test loader, batch size can be different if needed
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Data preprocessing complete.")

    # 3. Build Model
    print("\nBuilding LSTM model...")
    input_size = len(features)  # Number of features
    output_size = len(features) # Predicting all features

    model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LSTM_LAYERS, output_size, DROPOUT_PROB).to(device)
    print(model)

    # 4. Train Model
    print("\nTraining model...")
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_train_time = time.time()
    train_losses = []
    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            # Move data to the appropriate device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0) # Accumulate weighted loss

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.6f}')

        # Add validation loss calculation here using test_loader
        model.eval()
        with torch.no_grad():
           val_loss = 0.0
           for batch_X_val, batch_y_val in test_loader:
               batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
               outputs_val = model(batch_X_val)
               loss_val = criterion(outputs_val, batch_y_val)
               val_loss += loss_val.item() * batch_X_val.size(0)
           val_loss /= len(test_loader.dataset)
           print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}')


    end_train_time = time.time()
    print(f"Training finished in {end_train_time - start_train_time:.2f} seconds.")

    # Plot training loss and save it
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
    # Add validation loss plotting if you calculated it
    # plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss') # Uncomment if val_losses exists
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(OUTPUT_DIR, "training_loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"Training loss curve saved to {loss_plot_path}")
    plt.close() # Close the plot to free memory


    # 5. Prediction 
    print("\nPredicting future trends...")
    model.eval()  # Set model to evaluation mode

    # Use the last sequence from the original scaled data to start predictions
    last_sequence_scaled = scaled_data[-SEQ_LENGTH:]
    current_sequence = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dimension

    future_predictions_scaled = []

    with torch.no_grad(): # No need to track gradients during prediction
        for _ in range(N_FUTURE_STEPS):
            # Get the prediction for the next time step
            next_step_pred = model(current_sequence) # Shape: (1, output_size)
            future_predictions_scaled.append(next_step_pred.cpu().numpy().flatten()) # Store prediction

            # Update the sequence for the next prediction:
            # Remove the oldest time step and append the new prediction
            next_sequence_np = current_sequence.cpu().numpy().squeeze(0) # Shape: (seq_length, input_size)
            # Ensure the prediction has the correct shape before vstack
            pred_reshaped = next_step_pred.cpu().numpy().reshape(1, -1) # Reshape to (1, input_size)
            next_sequence_np = np.vstack((next_sequence_np[1:], pred_reshaped)) # Append prediction

            # Convert back to tensor and add batch dimension for the next loop iteration
            current_sequence = torch.tensor(next_sequence_np, dtype=torch.float32).unsqueeze(0).to(device)

    # Inverse transform the predictions to original scale
    future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled))
    print("Prediction complete.")

    # Create a DataFrame for the predictions
    last_timestamp = data_df.index[-1]
    future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=N_FUTURE_STEPS, freq='H')
    pred_df = pd.DataFrame(future_predictions, index=future_timestamps, columns=features)
    print("\nSample of Predicted Future Trends:")
    print(pred_df.head())

    # 6. Visualization and Saving Plots
    print("\nVisualizing results and saving plots...")
    n_plot_past = 200 # Number of past points to plot along with future predictions

    for feature in features:
        plt.figure(figsize=(15, 6))
        # Plot past data
        plt.plot(data_df.index[-n_plot_past:], data_df[feature].iloc[-n_plot_past:], label=f'Historical {feature}', color='blue')
        # Plot predicted future data
        plt.plot(pred_df.index, pred_df[feature], label=f'Predicted {feature}', color='red', linestyle='--')

        plt.title(f'{feature} - Historical Data and Future Prediction')
        plt.xlabel('Timestamp')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Save the plot
        plot_filename = f"{feature}_prediction_vs_historical.png"
        plot_path = os.path.join(OUTPUT_DIR, plot_filename)
        plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")
        plt.close() # Close the plot

    print("\nAnalysis complete.")