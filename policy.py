import torch.nn as nn


class PolicyNetwork(nn.Module):
    """La red que representa la politica parametrizada pi_theta

    Recibe un estado y devuelve un logit por cada accion posible (todavia sin
    pasar por softmax). La probabilidad de cada accion despues se obtiene como

        pi_theta(a|s) = softmax(logits)_a = e^(logit_a) / sum_a' e^(logit_a')

    pero esa parte se la deja a torch.distributions.Categorical(logits=...)
    que ya la aplica internamente y es mas estable numericamente que hacer 
    softmax a mano y despues log()

    La arquitectura es la que sugiere el enunciado
    """

    def __init__(self, input_dim, output_dim, hidden_dim=20, activation=nn.ELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, obs):
        return self.net(obs)
