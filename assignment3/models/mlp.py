from torch import nn

def init_weight(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    m.bias.data.zero_()

class TwoLayerMLP(nn.Module):
    """
    """
    def __init__(self, input_dim, output_dim, hidden_dim) -> None:
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )
        self.stack.apply(init_weight)


    def forward(self, inputs):
        """
        """
        return self.stack.forward(inputs)
