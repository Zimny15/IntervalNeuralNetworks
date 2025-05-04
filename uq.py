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
plt.plot(mean_pred, label = "Średnia predykcja")
plt.fill_between(np.arange(len(mean_pred)),
                 mean_pred - 2*std_pred,
                 mean_pred + 2*std_pred,
                 color='gray', alpha=0.4, label="Niepewność (±2 std)")
plt.legend()
plt.title("Niepewność z MCDrop")
plt.show()

#Abs Error
plt.figure(figsize=(14, 6))
absolute_error = np.abs(mean_pred - true_y)
uncertainty = 2 * std_pred
plt.fill_between(np.arange(len(mean_pred)),
                 0,
                 uncertainty,
                 color='cornflowerblue',
                 alpha=0.5,
                 label="Niepewność (±2 std)")

plt.plot(absolute_error, color='darkred', label='Absolute Error')
plt.title("Błąd bezwzględny i Niepewność (MC Dropout)")
plt.ylabel("Wartość")
plt.tight_layout()
plt.ylim(0, 1)
plt.show()



# ProbOut

class ProbOutDeconvNet(nn.Module):
    def __init__(self):
        super(ProbOutDeconvNet, self).__init__()
        layers = []
        channel_sizes = [1, 37, 74, 110, 147, 183, 220, 256, 171, 86, 2]  #teraz na wyjściu będzie mu i sigma
        dropout_pos = [3, 5, 7]
        for i in range(len(channel_sizes) - 1):
            in_c = channel_sizes[i]
            out_c = channel_sizes[i + 1]
            layers.append(nn.Conv1d(in_c, out_c, kernel_size=3, padding=1))
            if i < len(channel_sizes) - 2:
                layers.append(nn.ReLU())
            if i in dropout_pos:
                layers.append(nn.Dropout(p=0.5 if i > 3 else 0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def nll_loss(output, target):
    mu = output[:, 0:1, :]
    log_sigma = output[:, 1:2, :]
    sigma = torch.exp(log_sigma)
    loss = 0.5 * ((target - mu)**2 / (sigma**2)) + log_sigma
    return loss.mean()


def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs=100, patience=10):
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

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
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


if __name__ == "__main__":

    model_prob = ProbOutDeconvNet()
    optimizer = torch.optim.Adam(model_prob.parameters(), lr=1e-4)
    model_prob = train_model(model_prob, train_loader, val_loader, nll_loss, optimizer, epochs=100, patience=10)

    model_prob.eval()
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model_prob(xb)
            loss = nll_loss(pred, yb)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    torch.save(model_prob.state_dict(), "probout_model.pth")

    model_prob.eval()
    with torch.no_grad():
        x = x_test_tensor[0:1]  #próbka
        y = y_test_tensor[0:1]
        output = model_prob(x)
        mu = output[:, 0, :].squeeze().numpy()
        log_sigma = output[:, 1, :].squeeze().numpy()
        sigma = np.exp(log_sigma)
        true_y = y.squeeze().numpy()

        
        plt.figure(figsize=(12, 6))
        plt.plot(true_y, label="Prawdziwy sygnał")
        plt.plot(mu, label='Predykcja (μ)')
        plt.fill_between(np.arange(len(mu)), mu - 2*sigma, mu + 2*sigma, color='blue', alpha=0.4, label='Niepewność (σ)')
        plt.title("Niepewność z ProbOut")
        plt.legend()
        plt.tight_layout()
        plt.show()

        #Abs Error
        plt.figure(figsize=(14, 6))
        absolute_error = np.abs(mu - true_y)
        uncertainty = 2 * sigma
        plt.fill_between(np.arange(len(mu)),
                        0,
                        uncertainty,
                        color='cornflowerblue',
                        alpha=0.5,
                        label="Niepewność (±2 σ)")

        plt.plot(absolute_error, color='darkred', label='Absolute Error')
        plt.title("Błąd bezwzględny i Niepewność (ProbOut)")
        plt.ylabel("Wartość")
        plt.tight_layout()
        plt.ylim(0, 1)
        plt.show()
