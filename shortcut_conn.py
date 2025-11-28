import torch
import gpt_model


class ExampleDeepNeuralNetwork(torch.nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[0], layer_sizes[1]),
                    gpt_model.GELU(),
                ),
                torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[1], layer_sizes[2]), gpt_model.GELU()
                ),
                torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[2], layer_sizes[3]), gpt_model.GELU()
                ),
                torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[3], layer_sizes[4]), gpt_model.GELU()
                ),
                torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[4], layer_sizes[5]), gpt_model.GELU()
                ),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.0]])
    loss = torch.nn.MSELoss()
    loss = loss(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1.0, 0.0, -1.0]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)

print_gradients(model_without_shortcut, sample_input)

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)

print_gradients(model_with_shortcut, sample_input)
