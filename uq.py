import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from cnn import DeconvNet

model = DeconvNet()
model.load_state_dict(torch.load("base_model.pth"))
model.eval()


#Dane
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
x_sample = x_test_tensor[0].unsqueeze(0)  #(1, 1, 512)

mean_pred, std_pred = mc_dropout_predict(model, x_sample, T=64)

mean_pred = mean_pred.squeeze().numpy()
std_pred = std_pred.squeeze().numpy()
true_y = y_test_tensor[0].squeeze().numpy()

base_pred = model(x_sample).squeeze().detach().numpy()

#Generowanie wykresów 
plt.figure(figsize=(12, 6))
plt.plot(true_y, label="Prawdziwy sygnał")
plt.plot(mean_pred, label = "Predykcja modelu bazowego")
plt.fill_between(np.arange(len(mean_pred)),
                 mean_pred - 2*std_pred,
                 mean_pred + 2*std_pred,
                 color='gray', alpha=0.4, label="Niepewność (±2 std)")
plt.legend()
plt.title("Niepewność z MCDrop")
plt.show()


