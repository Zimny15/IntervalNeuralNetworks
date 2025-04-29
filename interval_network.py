import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class IntervalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(IntervalConv1d, self).__init__()
        self.weight_lower = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.weight_upper = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias_lower = nn.Parameter(torch.empty(out_channels))
        self.bias_upper = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_lower, a=5**0.5)
        nn.init.kaiming_uniform_(self.weight_upper, a=5**0.5)
        nn.init.zeros_(self.bias_lower)
        nn.init.zeros_(self.bias_upper)

    def forward(self, input_lower, input_upper):
        weight_min = torch.min(self.weight_lower, self.weight_upper)
        weight_max = torch.max(self.weight_lower, self.weight_upper)

        bias_min = torch.min(self.bias_lower, self.bias_upper)
        bias_max = torch.max(self.bias_lower, self.bias_upper)

        out_lower = nn.functional.conv1d(input_lower, torch.clamp(weight_min, min=0), bias=bias_min, padding=1) + \
                    nn.functional.conv1d(input_upper, torch.clamp(weight_max, max=0), bias=None, padding=1)

        out_upper = nn.functional.conv1d(input_upper, torch.clamp(weight_max, min=0), bias=bias_max, padding=1) + \
                    nn.functional.conv1d(input_lower, torch.clamp(weight_min, max=0), bias=None, padding=1)

        return out_lower, out_upper

class IntervalDeconvNet(nn.Module):
    def __init__(self):
        super(IntervalDeconvNet, self).__init__()
        channel_sizes = [1, 16, 32, 64, 128, 256, 128, 64, 32, 1]
        layers = []
        dropout_pos = [4, 7, 9]
        for i in range(len(channel_sizes) - 1):
            in_c = channel_sizes[i]
            out_c = channel_sizes[i+1]
            layers.append(IntervalConv1d(in_c, out_c, kernel_size=3, padding=1))
            
            # ReLU tylko jeśli to nie ostatnia warstwa
            if i < len(channel_sizes) - 2:
                layers.append(nn.ReLU())
            # Dropout w określonych miejscach
            if i in dropout_pos:
                layers.append(nn.Dropout(p=0.5 if i > 4 else 0.2))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        input_lower = input_upper = x
        for layer in self.net:
            if isinstance(layer, IntervalConv1d):
                input_lower, input_upper = layer(input_lower, input_upper)
            else:
                input_lower = layer(input_lower)
                input_upper = layer(input_upper)
        return input_lower, input_upper


def interval_loss(lower_pred, upper_pred, target, beta=0.002):
    zero = torch.zeros_like(target)
    lower_violation = torch.maximum(target - upper_pred, zero)
    upper_violation = torch.maximum(lower_pred - target, zero)
    penalty = beta * (upper_pred - lower_pred)
    return (lower_violation ** 2 + upper_violation ** 2 + penalty).mean()

def train_interval_net(model, train_loader, val_loader, epochs=100, beta=0.002, patience=10, lr=1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            output_lower, output_upper = model(xb)
            loss = interval_loss(output_lower, output_upper, yb, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                output_lower, output_upper = model(xb)
                loss = interval_loss(output_lower, output_upper, yb, beta=beta)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.6f}, Validation Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping!")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print("Training finished.")

def test_and_plot_intervals(model, test_loader, num_examples=5):
    model.eval()
    examples = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            lower, upper = model(xb)
            preds = (lower + upper) / 2
            interval_width = (upper - lower).squeeze(1)
            true = yb.squeeze(1)

            for i in range(xb.size(0)):
                if examples >= num_examples:
                    return
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 4))
                plt.plot(true[i].cpu().numpy(), label='True signal', color='black')
                plt.plot(preds[i].squeeze(0).cpu().numpy(), label='Prediction', color='blue')
                plt.fill_between(range(true.shape[-1]),
                                 lower[i].squeeze(0).cpu().numpy(),
                                 upper[i].squeeze(0).cpu().numpy(),
                                 color='blue', alpha=0.3, label='Interval')
                plt.title('Prediction with Interval Uncertainty')
                plt.legend()
                plt.show()
                examples += 1
