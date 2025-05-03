import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from cnn import DeconvNet  

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

# Trening
inn_model = IntervalDeconvNet(base_model)
optimizer = torch.optim.Adam(list(inn_model.lower_net.parameters()) + list(inn_model.upper_net.parameters()), lr=1e-5)

best_val_loss = float('inf')
patience, epochs_no_improve = 10, 0
best_state = None

for epoch in range(100):
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

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = inn_model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping")
            break

if best_state:
    inn_model.load_state_dict(best_state)

# Test
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

torch.save(inn_model.state_dict(), "inn_model.pth")

import matplotlib.pyplot as plt

# Wybór próbki testowej (np. indeks 0)
idx = 0
sample_x = x_test_tensor[idx:idx+1]  # batch dimension zachowany
true_y = y_test_tensor[idx].squeeze().cpu().numpy()


inn_model.eval()
with torch.no_grad():
    lower, upper = inn_model(sample_x)
    lower = lower.squeeze().cpu().numpy()
    upper = upper.squeeze().cpu().numpy()
    mean_pred = ((lower + upper) / 2)

# Wykresy
plt.figure(figsize=(14, 6))
plt.plot(true_y, label="Oryginalny sygnał (true)", color="blue")
plt.plot(mean_pred, label="Średnia predykcja", color="orange")
plt.fill_between(range(len(lower)), lower, upper, color="red", alpha=0.3, label="Przedział predykcji")

plt.title("Porównanie: sygnał vs predykcja INN")
plt.xlabel("Czas / Pozycja")
plt.ylabel("Wartość sygnału")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
