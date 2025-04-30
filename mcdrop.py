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

# 2. Normalizacja danych
x_mean = x_train_tensor.mean()
x_std = x_train_tensor.std()
y_mean = y_train_tensor.mean()
y_std = y_train_tensor.std()

x_train_tensor = (x_train_tensor - x_mean) / x_std
y_train_tensor = (y_train_tensor - y_mean) / y_std
x_val_tensor = (x_val_tensor - x_mean) / x_std
y_val_tensor = (y_val_tensor - y_mean) / y_std
x_test_tensor = (x_test_tensor - x_mean) / x_std
y_test_tensor = (y_test_tensor - y_mean) / y_std

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
#model_inn = IntervalDeconvNet()
#model_inn.load_state_dict(torch.load("interval_model.pth"))

# with torch.no_grad():
#     xb_sample, _ = next(iter(val_loader))
#     output_lower, output_upper = model_inn(xb_sample)
#     interval_width = (output_upper - output_lower).mean().item()
#     print(f"[Przed treningiem] Średnia szerokość przedziału (val): {interval_width:.6f}")


# train_interval_net(
#     model_inn,
#     train_loader,
#     val_loader,
#     epochs=20,     # ile dodatkowych epok
#     beta=5.0,      # zostaw to samo
#     patience=5,
#     lr=1e-4
# )
# 3. Zapisz z AKTUALIZACJĄ – pod nową nazwą lub nadpisz
#torch.save(model_inn.state_dict(), "interval_model_trained.pth")

# 2. Skopiuj wagi z DeconvNet do IntervalDeconvNet (bez torch.no_grad!)
# base_convs = [layer for layer in model.net if isinstance(layer, nn.Conv1d)]
# inn_convs = [layer for layer in model_inn.net if isinstance(layer, IntervalConv1d)]

# assert len(base_convs) == len(inn_convs), "Liczba warstw Conv1d nie zgadza się!"

# for layer_inn, layer_base in zip(inn_convs, base_convs):
#     w = layer_base.weight.data
#     layer_inn.weight_lower.data.copy_(w - 0.00005)
#     layer_inn.weight_upper.data.copy_(w + 0.00005)
#     b = layer_base.bias.data
#     layer_inn.bias_lower.data.copy_(b - 0.00005)
#     layer_inn.bias_upper.data.copy_(b + 0.00005)

# # 3. Upewnij się, że parametry są trenowalne
# for name, param in model_inn.named_parameters():
#     param.requires_grad = True

# # (opcjonalnie: diagnostyka)
# print("Przykładowy parametr i jego status gradientu:")
# for name, param in model_inn.named_parameters():
#     print(name, "-> requires_grad =", param.requires_grad)
#     break  # tylko jeden przykład

# # 4. Trenujemy szerokości przedziałów
# train_interval_net(
#     model_inn,
#     train_loader,
#     val_loader,
#     epochs=100,
#     beta=5.0,        # silna kara za szerokość
#     patience=5,
#     lr=1e-4
# )

# # Zapisz stan wytrenowanego modelu
# torch.save(model_inn.state_dict(), "interval_model2.pth")

# model_inn = IntervalDeconvNet()
# model_inn.load_state_dict(torch.load("interval_model2.pth"))
# # 7. Predykcja i odwrócenie normalizacji
# model_inn.eval()
# x_sample = x_test_tensor[0].unsqueeze(0)

# with torch.no_grad():
#     lower, upper = model_inn(x_sample)
#     lower = lower.squeeze().cpu() * y_std + y_mean
#     upper = upper.squeeze().cpu() * y_std + y_mean
#     mean_pred = ((lower + upper) / 2).numpy()
#     lower = lower.numpy()
#     upper = upper.numpy()
#     true_y = (y_test_tensor[0].squeeze() * y_std + y_mean).numpy()

# # 8. Wykres

# #plt.plot(true_y, label="Prawdziwy sygnał")
# plt.figure(figsize=(12, 6))
# plt.plot(mean_pred, label="Predykcja")
# plt.fill_between(range(len(mean_pred)), lower, upper,
#                  color='orange', alpha=0.4, label="Przedział niepewności")
# plt.legend()
# plt.title("Przedziałowa sieć wokół bazowej predykcji")
# plt.show()








import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from interval_network import IntervalDeconvNet  # <-- podmień na Twoją ścieżkę

# 1. Wczytaj model
model_inn = IntervalDeconvNet()
model_inn.load_state_dict(torch.load("interval_model2.pth"))
model_inn.eval()

# 2. Wczytaj dane testowe
x_test_tensor, y_test_tensor = torch.load("test_data.pth")

# 3. Normalizacja wejścia
x_mean = x_test_tensor.mean()
x_std = x_test_tensor.std()
y_mean = y_test_tensor.mean()
y_std = y_test_tensor.std()
y_mean, y_std = float(y_mean), float(y_std)

# 4. Próbka
idx = 0
x_sample = ((x_test_tensor[idx] - x_mean) / x_std).unsqueeze(0)

with torch.no_grad():
    lower, upper = model_inn(x_sample)
    lower = lower.squeeze().cpu().numpy()
    upper = upper.squeeze().cpu().numpy()

y_std = y_std.item() if isinstance(y_std, torch.Tensor) else y_std
y_mean = y_mean.item() if isinstance(y_mean, torch.Tensor) else y_mean

# 5. Denormalizacja predykcji
lower = lower * y_std + y_mean
upper = upper * y_std + y_mean
mean_pred = (lower + upper) / 2

# 6. (Opcjonalne) wygładzenie
mean_pred = gaussian_filter1d(mean_pred, sigma=1.0)
lower = gaussian_filter1d(lower, sigma=1.0)
upper = gaussian_filter1d(upper, sigma=1.0)

# 7. Wykres: tylko predykcja i przedziały
plt.figure(figsize=(10, 4))
plt.plot(mean_pred, label="Prediction", color='black', linewidth=1.5)
plt.fill_between(range(len(mean_pred)), lower, upper,
                 color='gray', alpha=0.3, label="Uncertainty interval")

plt.xlabel("Sample index", fontsize=11)
plt.ylabel("Signal value", fontsize=11)
plt.title("Interval prediction", fontsize=13)
plt.legend(frameon=False, fontsize=10)
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()