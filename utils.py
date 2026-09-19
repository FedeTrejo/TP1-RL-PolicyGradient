import numpy as np
import torch
from gymnasium import spaces


def get_input_dim(observation_space):
    """Cuantos numeros tiene que recibir la red de entrada, segun el tipo de espacio"""
    # Discrete: se representa como one-hot, asi que necesita un input por cada valor posible
    if isinstance(observation_space, spaces.Discrete):
        return int(observation_space.n)
    # Box: ya es un vector de floats, el input es directamente su dimension
    if isinstance(observation_space, spaces.Box):
        return int(np.prod(observation_space.shape))
    raise NotImplementedError(f"observation_space no soportado: {type(observation_space)}")


def get_output_dim(action_space):
    """La red va a tener una salida (un logit) por cada accion posible"""
    if isinstance(action_space, spaces.Discrete):
        return int(action_space.n)
    raise NotImplementedError(f"action_space no soportado: {type(action_space)}")


def obs_to_tensor(obs, observation_space, device="cpu"):
    """Pasa la observacion que devuelve el entorno a un tensor que la red pueda consumir

    Los entornos devuelven un entero (Discrete) y un vector de floats (Box). La red 
    necesita un vector de tamano fijo en los dos casos, asi que a la observacion discreta 
    se la codifica como one-hot (ej. si obs=3 en un Discrete(6, start=1), queda [0,1,0,0,0,0]) 
    y a la continua se la deja tal cual viene
    """
    if isinstance(observation_space, spaces.Discrete):
        index = int(obs) - observation_space.start
        one_hot = torch.zeros(observation_space.n, dtype=torch.float32, device=device)
        one_hot[index] = 1.0
        return one_hot
    if isinstance(observation_space, spaces.Box):
        return torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
    raise NotImplementedError(f"observation_space no soportado: {type(observation_space)}")


def discounted_returns(rewards, gamma):
    """Dada la lista de recompensas de una trayectoria, devuelve el retorno descontado en cada paso

    Para el paso t esto es R_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    No hace falta sumar todo eso de nuevo en cada t: yendo de atras para
    adelante, el retorno en t es simplemente la recompensa de ese paso mas
    gamma por el retorno que ya se calculo para t+1 (R_t = r_t + gamma * R_{t+1}).
    Por eso se recorre rewards en reversa, arrastrando ese acumulado
    """
    returns = [0.0] * len(rewards)
    R = 0.0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        returns[t] = R
    return returns
