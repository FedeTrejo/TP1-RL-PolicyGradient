import torch.nn as nn


class ValueNetwork(nn.Module):
    """Aproxima V(s): el retorno esperado a partir del estado s siguiendo la
    politica actual. Se usa como baseline para restarle
    varianza al estimador del gradiente de politica

    Arquitectura mas chica que la de la politica (una sola capa oculta): alcanza 
    con menos capacidad para aproximar un solo numero por estado
    """

    def __init__(self, input_dim, hidden_dim=20, activation=nn.ELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs)
