"""Barrido de semillas: entrena el mismo experimento con varias semillas y
devuelve todos los historiales, para poder reportar el promedio en vez de una
corrida suelta

Una funcion definida en una celda del notebook no se puede picklear, asi que 
el worker tiene que estar en un .py

Por que en paralelo: las redes son de 20 neuronas, o sea que el cuello de
botella es el overhead de Python por paso. Así las cinco semillas salen
casi al mismo costo de que una sola en vez de cinco veces
"""

import contextlib
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import gymnasium as gym

from envs import (TwoAZeroObsOneStep, TwoARandomObsOneStep,
                  LineWorldEasyEnv, LineWorldMirrorEnv)

SEMILLAS = [0, 1, 2, 3, 4]

# El worker recibe el nombre del entorno, no el entorno: los entornos de
# gymnasium no se pueden picklear para mandarlos a otro proceso
REGISTRO = {
    "TwoAZeroObsOneStep": TwoAZeroObsOneStep,
    "TwoARandomObsOneStep": TwoARandomObsOneStep,
    "LineWorldEasyEnv": LineWorldEasyEnv,
    "LineWorldMirrorEnv": LineWorldMirrorEnv,
    "CartPole-v1": lambda: gym.make("CartPole-v1"),
    "Acrobot-v1": lambda: gym.make("Acrobot-v1"),
}


def crear_entorno(nombre, semilla=None):
    """Arma el entorno y si se le da semilla la fija en los dos generadores
    que usa: el de la dinamica (reset) y el del action_space (sample).
    El segundo es independiente del primero y es el que usa el agente
    aleatorio de referencia: sin fijarlo, esa linea cambia entre corridas"""
    env = REGISTRO[nombre]()
    if semilla is not None:
        env.reset(seed=semilla)
        env.action_space.seed(semilla)
    return env


def _entrenar(args):
    """Worker: corre una semilla y devuelve cosas picklables (el historial y los
    pesos)"""
    (nombre, semilla, n_iterations, batch_size, gamma,
     policy_lr, con_baseline, value_lr) = args

    # cada proceso con un hilo
    torch.set_num_threads(1)

    if con_baseline:
        from reinforce_baseline import REINFORCEBaselineAgent, train as entrenar
    else:
        from reinforce import REINFORCEAgent, train as entrenar

    env = crear_entorno(nombre, semilla)
    torch.manual_seed(semilla)
    np.random.seed(semilla)

    if con_baseline:
        agente = REINFORCEBaselineAgent(env.observation_space, env.action_space,
                                        policy_lr=policy_lr, value_lr=value_lr)
    else:
        agente = REINFORCEAgent(env.observation_space, env.action_space, lr=policy_lr)

    with open(os.devnull, "w") as mudo, contextlib.redirect_stderr(mudo):
        historial = entrenar(env, agente, n_iterations=n_iterations,
                             batch_size=batch_size, gamma=gamma,
                             progress_desc=f"{nombre} s{semilla}")
    env.close()

    if con_baseline:
        pesos = {"policy": agente.policy.state_dict(),
                 "value": agente.value_net.state_dict()}
    else:
        pesos = agente.policy.state_dict()

    return {"semilla": semilla, "history": historial, "pesos": pesos}


def barrido(nombre, n_iterations, batch_size, gamma, policy_lr=1e-2,
            con_baseline=False, value_lr=1e-2, semillas=None):
    """Entrena nombre una vez por semilla en paralelo y devuelve la lista de
    resultados ordenada por semilla"""
    semillas = SEMILLAS if semillas is None else semillas
    tareas = [(nombre, s, n_iterations, batch_size, gamma,
               policy_lr, con_baseline, value_lr) for s in semillas]
    with ProcessPoolExecutor(max_workers=len(tareas)) as pool:
        salidas = list(pool.map(_entrenar, tareas))
    return sorted(salidas, key=lambda r: r["semilla"])


def apilar(resultados, clave):
    """Pone la misma metrica de todas las semillas en una matriz
    (semillas x trayectorias) recortando a la longitud comun"""
    series = [np.asarray(r["history"][clave], dtype=float) for r in resultados]
    largo = min(len(s) for s in series)
    return np.vstack([s[:largo] for s in series])


def promedio_y_desvio(resultados, clave):
    matriz = apilar(resultados, clave)
    return matriz.mean(axis=0), matriz.std(axis=0), matriz
