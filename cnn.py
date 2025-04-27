import numpy as np
from scipy.fftpack import dct #Dyskretna transformacja kosinusowa
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from interval_network import IntervalDeconvNet, train_interval_net, test_and_plot_intervals

#Generowanie danych
def generate_signal(length=512, num_jumps=7, min_val=0, max_val=1):

    jump_positions = np.sort(np.random.choice(range(1, length-1), num_jumps, replace=False))

    segment_boundaries = np.concatenate(([0], jump_positions, [length]))

    segment_values = np.random.uniform(min_val, max_val, size=len(segment_boundaries)-1)

    signal = np.zeros(length)
    for i in range(len(segment_values)):
        start = segment_boundaries[i]
        end = segment_boundaries[i+1]
        signal[start:end] = segment_values[i]
    
    return signal

def blur_signal(y):
    length = len(y)
    D = dct(np.eye(length), norm='ortho')  
    decay = np.exp(-np.linspace(0, 5, length))  #Wartości malejące wykładniczo
    S = np.diag(decay)
    A = D.T @ S @ D
    x = A @ y
    x = x + np.random.normal(0, 0.05, size=x.shape) #Szum
    return x

#Chcemy wygenerować macierz danych o wymiarach 2000 x 512 (2000 wektorów o 512 wartościach tworzących sygnał)
num_samples = 2000
signal_length = 512

y_matrix = np.empty((num_samples, signal_length))
x_matrix = np.empty((num_samples, signal_length))

for i in range(num_samples):
    y = generate_signal(length=signal_length)
    x = blur_signal(y)
    y_matrix[i] = y
    x_matrix[i] = x

#Tworzenie sieci 
x_train, y_train = x_matrix[:1600], y_matrix[:1600]
x_val, y_val = x_matrix[1600:1800], y_matrix[1600:1800]
x_test, y_test = x_matrix[1800:], y_matrix[1800:]

class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()
        
        layers = []
        channel_sizes = [1, 16, 32, 64, 128, 256, 128, 64, 32, 1]
        dropout_pos = [4, 7, 9]

        for i in range(len(channel_sizes) - 1):
            in_c = channel_sizes[i]
            out_c = channel_sizes[i+1]

            layers.append(nn.Conv1d(in_c, out_c, kernel_size=3, padding=1))
            
            #ReLU tylko jeśli to nie ostatnia warstwa
            if i < len(channel_sizes) - 2:
                layers.append(nn.ReLU())

            #Dropout w określonych miejscach
            if i in dropout_pos:
                layers.append(nn.Dropout(p=0.5 if i > 4 else 0.2))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

#Zmieniamy dane wejściowe na tensory i tworzymy DataLoader
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

x_val_tensor = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

batch_size = 256
train_ds = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

val_ds = TensorDataset(x_val_tensor, y_val_tensor)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

test_ds = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# model przedzialowy

model_inn = IntervalDeconvNet()

# Trening modelu
train_interval_net(
    model_inn, 
    train_loader, 
    val_loader, 
    epochs=100,    # liczba epok
    beta=0.002,    # współczynnik szerokości przedziału
    patience=10,   # early stopping
    lr=1e-5        # mały learning rate (jak w artykule)
)

# Testowanie i wizualizacja
test_and_plot_intervals(
    model_inn, 
    test_loader, 
    num_examples=5   # liczba wykresów do narysowania
)




model = DeconvNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

#Early stopping
patience = 10  #Liczba epok bez poprawy
best_val_loss = float('inf')  
epochs_without_improvement = 0  
best_model_state = None  #Najlepszy model

for epoch in range(100):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    #Obliczanie straty na zbiorze walidacyjnym
    model.eval() 
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        best_model_state = model.state_dict()  #Zapis najlepszego modelu
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

#Testowanie modelu 
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

test_ds = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

model.eval()
test_loss = 0
with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")


#Ocena niepewności MCDrop
def mc_dropout_predict(model, x, T=64):
    
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()  

    preds = []
    with torch.no_grad():
        for _ in range(T):
            pred = model(x)
            preds.append(pred.unsqueeze(0))  #dodaj wymiar próbki na początku

    preds = torch.cat(preds, dim=0)  #(T, batch_size, 1, length)
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0)
    return mean_pred, std_pred

#Próbka do wykresów 
x_sample = x_test_tensor[0].unsqueeze(0)  #(1, 1, 512)

mean_pred, std_pred = mc_dropout_predict(model, x_sample, T=100)

mean_pred = mean_pred.squeeze().cpu().numpy()
std_pred = std_pred.squeeze().cpu().numpy()
true_y = y_test_tensor[0].squeeze().cpu().numpy()

plt.figure(figsize=(12, 6))
plt.plot(true_y, label="Prawdziwy sygnał")
plt.plot(mean_pred, label="Średnia predykcja")
plt.fill_between(np.arange(len(mean_pred)),
                 mean_pred - 2*std_pred,
                 mean_pred + 2*std_pred,
                 color='gray', alpha=0.4, label="Niepewność (±2 std)")
plt.legend()
plt.title("Niepewność z MCDrop")
plt.show()