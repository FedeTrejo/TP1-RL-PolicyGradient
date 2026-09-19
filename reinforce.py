import torch
from torch.distributions import Categorical
from tqdm import trange

from policy import PolicyNetwork
from utils import get_input_dim, get_output_dim, obs_to_tensor, discounted_returns


class REINFORCEAgent:
    """Junta la red de politica con su optimizador y expone lo que hace falta para
    entrenar (muestrear acciones, quedandose con el log-prob) y para evaluar
    (la accion mas probable, sin muestrear)"""

    def __init__(self, observation_space, action_space, hidden_dim=20, lr=1e-2, device="cpu"):
        self.observation_space = observation_space
        self.device = device
        input_dim = get_input_dim(observation_space)
        output_dim = get_output_dim(action_space)
        self.policy = PolicyNetwork(input_dim, output_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def _logits(self, obs):
        # f_theta(s): forward de la red todavia sin softmax
        state = obs_to_tensor(obs, self.observation_space, self.device)
        return self.policy(state)

    def act(self, obs):
        """Muestrea a_t de pi_theta(.|s_t). Politica estocástica: sample(), no argmax

        Categorical(logits=...) arma la distribucion aplicando softmax internamente,
        y dist.log_prob(action) da log(pi_theta(a_t|s_t)) directamente, ya con
        grafo de autograd (necesario intacto para el backward, una vez juntado
        todo el batch de trayectorias)
        """
        logits = self._logits(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def act_deterministic(self, obs):
        """La accion de mayor probabilidad segun la politica. Solo para evaluar
        al agente ya entrenado, nunca durante el entrenamiento"""
        with torch.no_grad():
            action = torch.argmax(self._logits(obs))
        return int(action.item())

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))


def collect_batch(env, agent, batch_size, gamma):
    """Corre batch_size trayectorias completas con la politica actual y junta todo
    lo que hace falta para estimar el gradiente

    Por cada trayectoria se guarda el log-prob de cada accion tomada y el retorno
    descontado R(t) de cada paso y al final se concatena todo en un solo tensor

    Ademas se devuelve por separado el retorno total y el largo de cada trayectoria
    individual: eso no se usa para entrenar, es solo para poder graficar despues
    como fue evolucionando el desempeno trayectoria a trayectoria
    """
    batch_log_probs = []
    batch_returns = []
    episode_returns = []
    episode_lengths = []

    for _ in range(batch_size):
        obs, _ = env.reset()
        log_probs, rewards = [], []
        terminated = truncated = False

        # Una trayectoria completa hasta terminar
        while not (terminated or truncated):
            action, log_prob = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)

        batch_log_probs.extend(log_probs)
        batch_returns.extend(discounted_returns(rewards, gamma))
        episode_returns.append(sum(rewards))
        episode_lengths.append(len(rewards))

    log_probs_tensor = torch.stack(batch_log_probs)
    returns_tensor = torch.tensor(batch_returns, dtype=torch.float32, device=agent.device)
    return log_probs_tensor, returns_tensor, episode_returns, episode_lengths


def reinforce_update(agent, log_probs, returns, batch_size):
    """Un paso de entrenamiento: arma la loss a partir del batch recolectado y
    hace backward + step de Adam

    El gradiente que hay que seguir es:

        g_hat = (1/N) * sum_i sum_t  grad_theta[log pi_theta(a_t|s_t)] * R(t)

    y eso es lo que sale de derivar log_probs.sum()/batch_size respecto a
    theta si R(t) se trata como una constante (no depende de theta). Como hace
    falta ascenso de gradiente pero Adam hace descenso, la loss que se pasa
    es con - de esa expresion: minimizar -g_hat equivale a maximizar g_hat

    Antes de eso se normalizan los retornos del batch (media 0, desvio 1): reduce
    mucho la varianza del estimador y evita que un entorno con retornos en
    otra escala rompa el learning rate
    """
    normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = -(log_probs * normalized_returns).sum() / batch_size

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()
    return loss.item()


def train(env, agent, n_iterations, batch_size, gamma, writer=None, progress_desc="REINFORCE"):
    """El loop de entrenamiento: en cada iteracion junta un batch de
    trayectorias con la politica actual y hace un update con ellas

    El historial devuelto tiene una entrada por trayectoria (no por batch), que es
    lo que despues se grafica como recompensa por trayectoria. La loss en cambio
    es una sola cifra por batch (es una unica actualizacion conjunta sobre todas
    esas trayectorias), asi que al loguearlo se repite el mismo valor para cada
    trayectoria de ese batch
    """
    history = {"episode": [], "episode_return": [], "episode_length": [], "loss": []}
    episode_counter = 0

    # se capa a mano cuantas veces se actualiza el texto de la barra de tqdm 
    # independientemente de cuantas iteraciones haya
    report_every = max(1, n_iterations // 20)
    pbar = trange(n_iterations, desc=progress_desc, mininterval=1e9)
    for it in pbar:
        log_probs, returns, ep_returns, ep_lengths = collect_batch(env, agent, batch_size, gamma)
        loss = reinforce_update(agent, log_probs, returns, batch_size)

        for ep_return, ep_length in zip(ep_returns, ep_lengths):
            history["episode"].append(episode_counter)
            history["episode_return"].append(ep_return)
            history["episode_length"].append(ep_length)
            history["loss"].append(loss)

            if writer is not None:
                writer.add_scalar("Performance/EpisodeReturn", ep_return, episode_counter)
                writer.add_scalar("Performance/EpisodeLength", ep_length, episode_counter)
                writer.add_scalar("Loss/train", loss, episode_counter)

            episode_counter += 1

        if it % report_every == 0 or it == n_iterations - 1:
            avg_return = sum(ep_returns) / len(ep_returns)
            pbar.n = it + 1
            pbar.set_postfix(avg_return=f"{avg_return:.2f}", loss=f"{loss:.3f}", refresh=False)
            pbar.refresh()

    return history
