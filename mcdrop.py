import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from cnn import DeconvNet
from interval_network import IntervalDeconvNet, IntervalConv1d, train_interval_net

model = DeconvNet()
model.load_state_dict(torch.load("base_model.pth"))
model.eval()


# 2. Wczytaj dane treningowe, walidacyjne i testowe
x_train_tensor, y_train_tensor = torch.load("train_data.pth")
x_val_tensor, y_val_tensor = torch.load("val_data.pth")
x_test_tensor, y_test_tensor = torch.load("test_data.pth")

batch_size = 256

train_ds = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

val_ds = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

test_ds = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

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
x_test_tensor, y_test_tensor = torch.load("test_data.pth")

x_sample = x_test_tensor[0].unsqueeze(0)  #(1, 1, 512)

mean_pred, std_pred = mc_dropout_predict(model, x_sample, T=64)

mean_pred = mean_pred.squeeze().numpy()
std_pred = std_pred.squeeze().numpy()
true_y = y_test_tensor[0].squeeze().numpy()

#Generowanie wykresów 
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


# 1. Stwórz model przedziałowy
model_inn = IntervalDeconvNet()

# 2. Skopiuj wagi z DeconvNet do IntervalDeconvNet (bez torch.no_grad!)
base_convs = [layer for layer in model.net if isinstance(layer, nn.Conv1d)]
inn_convs = [layer for layer in model_inn.net if isinstance(layer, IntervalConv1d)]

assert len(base_convs) == len(inn_convs), "Liczba warstw Conv1d nie zgadza się!"

for layer_inn, layer_base in zip(inn_convs, base_convs):
    layer_inn.weight_lower.data.copy_(layer_base.weight.data)
    layer_inn.weight_upper.data.copy_(layer_base.weight.data)
    layer_inn.bias_lower.data.copy_(layer_base.bias.data)
    layer_inn.bias_upper.data.copy_(layer_base.bias.data)

# 3. Upewnij się, że parametry są trenowalne
for name, param in model_inn.named_parameters():
    param.requires_grad = True

# (opcjonalnie: diagnostyka)
print("Przykładowy parametr i jego status gradientu:")
for name, param in model_inn.named_parameters():
    print(name, "-> requires_grad =", param.requires_grad)
    break  # tylko jeden przykład

# 4. Trenujemy szerokości przedziałów
train_interval_net(
    model_inn,
    train_loader,
    val_loader,
    epochs=30,
    beta=0.05,
    patience=5,
    lr=1e-3
)

# 5. Predykcja i wykres dla jednej próbki
model_inn.eval()

x_sample = x_test_tensor[0].unsqueeze(0)  # (1, 1, 512)

with torch.no_grad():
    lower, upper = model_inn(x_sample)
    lower = lower.squeeze().cpu().numpy()
    upper = upper.squeeze().cpu().numpy()
    mean_pred = (lower + upper) / 2
    interval_size = upper - lower

true_y = y_test_tensor[0].squeeze().numpy()

# 6. Generowanie wykresu
plt.figure(figsize=(12, 6))
plt.plot(true_y, label="Prawdziwy sygnał")
plt.plot(mean_pred, label="Predykcja")
plt.fill_between(np.arange(len(mean_pred)),
                 lower,
                 upper,
                 color='orange', alpha=0.4, label="Przedział niepewności")
plt.legend()
plt.title("Przedziałowa sieć wokół bazowej predykcji")
plt.show()