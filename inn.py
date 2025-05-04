import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from cnn import DeconvNet  
import matplotlib.pyplot as plt
import numpy as np

# Bazowy modelu + dane
base_model = DeconvNet()
base_model.load_state_dict(torch.load("base_model.pth"))
base_model.eval()

x_train_tensor, y_train_tensor = torch.load("train_data.pth")
x_val_tensor, y_val_tensor = torch.load("val_data.pth")
x_test_tensor, y_test_tensor = torch.load("test_data.pth")

batch = 256

train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=batch)
test_loader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=batch)

# Definicja sieci przedziałowej
class IntervalDeconvNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.lower_net = DeconvNet()
        self.upper_net = DeconvNet()
        self.lower_net.load_state_dict(base_model.state_dict())
        self.upper_net.load_state_dict(base_model.state_dict())
        
        for param in base_model.parameters(): # Zamrażanie oryginalnych parametrów
            param.requires_grad = False

    def forward(self, x):
        lower = self.lower_net(x)
        upper = self.upper_net(x)
        return lower, upper

# Funkcja straty dla INN
def inn_loss_fn(lower, upper, target, beta=0.002):
    zero = torch.zeros_like(target)
    lower_violation = torch.maximum(lower - target, zero)
    upper_violation = torch.maximum(target - upper, zero)
    interval_width = upper - lower
    loss = (lower_violation**2 + upper_violation**2).mean() + beta * interval_width.mean()
    return loss

import torch

def train_interval_model(base_model, train_loader, val_loader, test_loader, inn_loss_fn, 
                         learning_rate=1e-5, max_epochs=100, patience=10, model_path="inn_model.pth"):
    # Inicjalizacja modelu i optymalizatora
    inn_model = IntervalDeconvNet(base_model)
    optimizer = torch.optim.Adam(
        list(inn_model.lower_net.parameters()) + list(inn_model.upper_net.parameters()),
        lr=learning_rate
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None

    # Trening i walidacja
    for epoch in range(max_epochs):
        inn_model.train()
        total_loss = 0
        for xb, yb in train_loader:
            lower, upper = inn_model(xb)
            loss = inn_loss_fn(lower, upper, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Walidacja
        inn_model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                lower, upper = inn_model(xb)
                val_loss += inn_loss_fn(lower, upper, yb).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {total_loss / len(train_loader):.4f}, Val Loss = {val_loss:.4f}")

        # Wczesne zatrzymanie
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = inn_model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

    # najlepszy stan modelu
    if best_state:
        inn_model.load_state_dict(best_state)

    # Testowanie modelu
    inn_model.eval()
    test_loss = 0
    total, covered = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            lower, upper = inn_model(xb)
            test_loss += inn_loss_fn(lower, upper, yb).item()
            is_covered = (yb >= lower) & (yb <= upper)
            covered += is_covered.sum().item()
            total += yb.numel()

    test_loss /= len(test_loader)
    coverage = covered / total

    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Coverage: {coverage:.2%}")

    # Zapisz model
    torch.save(inn_model.state_dict(), model_path)

    return inn_model

if __name__ == "__main__":
    inn_model = train_interval_model(base_model, train_loader, val_loader, test_loader, inn_loss_fn)
    # Wykresy
    idx = 0
    sample_x = x_test_tensor[idx:idx+1]  
    true_y = y_test_tensor[idx].squeeze().cpu().numpy()
    
    inn_model.eval()
    with torch.no_grad():
        lower, upper = inn_model(sample_x)
        lower = lower.squeeze().cpu().numpy()
        upper = upper.squeeze().cpu().numpy()
        mean_pred = ((lower + upper) / 2)


    plt.figure(figsize=(12, 6))
    plt.plot(true_y, label="Prawdziwy sygnał")
    plt.plot(mean_pred, label="Średnia predykcja")
    plt.fill_between(range(len(lower)), lower, upper, color="grey", alpha=0.4, label="Przedział predykcji")

    plt.title("Niepewność z INN")
    plt.ylabel("Wartość sygnału")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #Abs Error
    plt.figure(figsize=(12, 6))
    absolute_error = np.abs(mean_pred - true_y)
    uncertainty = upper - lower
    plt.fill_between(np.arange(len(mean_pred)),
                    0,
                    uncertainty,
                    color='cornflowerblue',
                    alpha=0.5,
                    label="inn_un")

    plt.plot(absolute_error, color='darkred', label='Absolute Error')
    plt.title("Błąd bezwzględny i Niepewność (INN)")
    plt.ylabel("Wartość")
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.show()

