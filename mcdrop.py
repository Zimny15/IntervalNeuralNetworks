import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from cnn import DeconvNet
from interval_network import IntervalDeconvNet, IntervalConv1d, train_interval_net

model = DeconvNet()
model.load_state_dict(torch.load("base_model.pth"))
model.eval()

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


# Tworzymy nową sieć przedziałową
model_inn = IntervalDeconvNet()

# Kopiujemy wytrenowane parametry DeconvNet -> IntervalDeconvNet
with torch.no_grad():
    for layer_inn, layer_base in zip(model_inn.net, model.net):
        if isinstance(layer_inn, IntervalConv1d) and isinstance(layer_base, nn.Conv1d):
            layer_inn.weight_lower.copy_(layer_base.weight)
            layer_inn.weight_upper.copy_(layer_base.weight)
            layer_inn.bias_lower.copy_(layer_base.bias)
            layer_inn.bias_upper.copy_(layer_base.bias)

# UWAGA! Potrzebujesz danych treningowych do trenowania szerokości przedziałów
# Jeśli nie masz zapisanych x_train_tensor, y_train_tensor -> musisz je wczytać z pliku lub wygenerować
# Załaduj train_loader, val_loader odpowiednio!

# Trenujemy szerokości przedziałów
train_interval_net(
    model_inn, 
    train_loader,    # <- musi być zdefiniowany wcześniej!
    val_loader,      # <- musi być zdefiniowany wcześniej!
    epochs=30,       
    beta=0.002,      
    patience=5,
    lr=1e-6
)

# Predykcja IntervalNet na tej samej próbce
model_inn.eval()
x_sample = x_test_tensor[0].unsqueeze(0)

with torch.no_grad():
    lower, upper = model_inn(x_sample)
    lower = lower.squeeze().cpu().numpy()
    upper = upper.squeeze().cpu().numpy()
    mean_pred = (lower + upper) / 2
    interval_size = upper - lower

true_y = y_test_tensor[0].squeeze().numpy()

# Wykres przedziałowej sieci
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