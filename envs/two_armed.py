import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TwoAZeroObsOneStep(gym.Env):
    """Bandit de un paso sin nada de estado real

    Accion 0 -> reward -1 (la "mala")
    Accion 1 -> reward +1 (la "buena")
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # observacion dummy: siempre 0 no aporta ninguna informacion
        self.observation_space = spaces.Discrete(1)
        # 2 palancas posibles
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # configura self.np_random con la semilla
        observation = 0
        info = {}
        return observation, info

    def step(self, action):
        assert self.action_space.contains(action), f"accion invalida: {action}"
        reward = 1.0 if action == 1 else -1.0
        observation = 0
        terminated = True  # un solo paso y listo
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info


class TwoARandomObsOneStep(gym.Env):
    """Bandit contextual de un paso: ahora si hay dos estados posibles
    y la accion correcta depende del estado en el que arranca la trayectoria

    Estado S1 (obs=0): accion 0 -> -1, accion 1 -> +1
    Estado S2 (obs=1): accion 0 -> +1, accion 1 -> -1
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)
        self._state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # la trayectoria arranca en S1 o S2 con la misma probabilidad
        self._state = int(self.np_random.integers(0, 2))
        info = {}
        return self._state, info

    def step(self, action):
        assert self.action_space.contains(action), f"accion invalida: {action}"
        good_action = 1 if self._state == 0 else 0
        reward = 1.0 if action == good_action else -1.0
        observation = self._state
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info
