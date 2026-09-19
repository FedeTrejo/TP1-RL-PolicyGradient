import gymnasium as gym
from gymnasium import spaces

LEFT = 0
RIGHT = 1


class LineWorldEasyEnv(gym.Env):
    """Pasillo de 6 casilleros, posicion inicial en el extremo izquierdo

    Observacion: posicion actual, entero en [1, 6].
    Acciones: 0 = izquierda, 1 = derecha.
    Recompensa: +1 al alcanzar la posicion 6 (objetivo), 0 en otro caso.
    La trayectoria termina al alcanzar el objetivo
    """

    metadata = {"render_modes": []}
    N_POSITIONS = 6

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(self.N_POSITIONS, start=1)
        self.action_space = spaces.Discrete(2)
        self._pos = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = 1
        info = {}
        return self._pos, info

    def step(self, action):
        assert self.action_space.contains(action), f"accion invalida: {action}"
        if action == RIGHT:
            self._pos = min(self._pos + 1, self.N_POSITIONS)
        else:
            self._pos = max(self._pos - 1, 1)

        terminated = self._pos == self.N_POSITIONS
        reward = 1.0 if terminated else 0.0
        truncated = False
        info = {}
        return self._pos, reward, terminated, truncated, info


class LineWorldMirrorEnv(gym.Env):
    """Pasillo de 4 estados

    Observacion: posicion actual, entero en [1, 4]. La posicion 4 es el objetivo.
    Acciones: 0 = izquierda, 1 = derecha, excepto en la posicion 2 donde
    izquierda mueve a la derecha y viceversa.
    Recompensa: -1 en cada paso.
    La trayectoria termina al alcanzar el objetivo (posicion 4)
    """

    metadata = {"render_modes": []}
    N_POSITIONS = 4
    MIRROR_POSITION = 2

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(self.N_POSITIONS, start=1)
        self.action_space = spaces.Discrete(2)
        self._pos = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = 1
        info = {}
        return self._pos, info

    def step(self, action):
        assert self.action_space.contains(action), f"accion invalida: {action}"

        effective_action = action
        if self._pos == self.MIRROR_POSITION:
            effective_action = RIGHT if action == LEFT else LEFT

        if effective_action == RIGHT:
            self._pos = min(self._pos + 1, self.N_POSITIONS)
        else:
            self._pos = max(self._pos - 1, 1)

        terminated = self._pos == self.N_POSITIONS
        reward = -1.0
        truncated = False
        info = {}
        return self._pos, reward, terminated, truncated, info
