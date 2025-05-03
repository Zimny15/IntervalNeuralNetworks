import numpy as np
from scipy.fftpack import dct #Dyskretna transformacja kosinusowa
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Definicja modelu 
class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()
        
        layers = []
        channel_sizes = [1, 37, 74, 110, 147, 183, 220, 256, 171, 86, 1]
        dropout_pos = [3, 5, 7]

        for i in range(len(channel_sizes) - 1):
            in_c = channel_sizes[i]
            out_c = channel_sizes[i+1]

            layers.append(nn.Conv1d(in_c, out_c, kernel_size=3, padding=1))
            
            #ReLU tylko jeśli to nie ostatnia warstwa
            if i < len(channel_sizes) - 2:
                layers.append(nn.ReLU())

            #Dropout w określonych miejscach
            if i in dropout_pos:
                layers.append(nn.Dropout(p=0.5 if i > 3 else 0.2))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Generowanie danych
def generate_signal(length=512, num_jumps=20, min_val=-1, max_val=1):

    jump_positions = np.sort(np.random.choice(range(1, length-1), num_jumps, replace=False))

    segment_boundaries = np.concatenate(([0], jump_positions, [length]))

    segment_values = np.random.uniform(min_val, max_val, size=len(segment_boundaries)-1)

    signal = np.zeros(length)
    for i in range(len(segment_values)):
        start = segment_boundaries[i]
        end = segment_boundaries[i+1]
        signal[start:end] = segment_values[i]
    
    return signal

def blur_signal(y, noise = True):
    length = len(y)
    D = dct(np.eye(length), norm='ortho')  
    decay = np.exp(-np.linspace(0, 5, length))  #Wartości malejące wykładniczo
    S = np.diag(decay)
    A = D.T @ S @ D
    x = A @ y
    if noise:
        g_noise = np.random.normal(0, 0.05, size=y.shape)
        x = x + g_noise
        y = y + g_noise
    return x, y


def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=100, patience=10):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Walidacja
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


if __name__ == "__main__":

    #Chcemy wygenerować macierz danych o wymiarach 2000 x 512 (2000 wektorów o 512 wartościach tworzących sygnał)
    num_samples = 2000
    signal_length = 512
    y_matrix = np.empty((num_samples, signal_length))
    x_matrix = np.empty((num_samples, signal_length))

    for i in range(num_samples):
        y = generate_signal(length=signal_length)
        x, y = blur_signal(y)
        y_matrix[i] = y 
        x_matrix[i] = x

    #Tworzenie sieci 
    x_train, y_train = x_matrix[:1600], y_matrix[:1600]
    x_val, y_val = x_matrix[1600:1800], y_matrix[1600:1800]
    x_test, y_test = x_matrix[1800:], y_matrix[1800:]

    #Zmieniamy dane wejściowe na tensory i tworzymy DataLoader
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    torch.save((x_train_tensor, y_train_tensor), "train_data.pth")

    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    torch.save((x_val_tensor, y_val_tensor), "val_data.pth")

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    torch.save((x_test_tensor, y_test_tensor), "test_data.pth")

    batch_size = 256

    train_ds = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    test_ds = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    #Trenowanie modelu 
    model = DeconvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model = train_model(model, train_loader, val_loader, loss_fn, optimizer,
                              epochs=100, patience=10)
    
    #Testowanie modelu 
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            test_loss += loss.item()
            test_loss /= len(test_loader)
            print(f"Test Loss: {test_loss:.4f}")
    #Zapis modelu 
    torch.save(model.state_dict(), "base_model.pth")

