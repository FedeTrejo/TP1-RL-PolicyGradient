import torch
from tqdm import trange

from reinforce import REINFORCEAgent
from value import ValueNetwork
from utils import get_input_dim, obs_to_tensor, discounted_returns


class REINFORCEBaselineAgent(REINFORCEAgent):
    """El mismo agente de la Parte 2 (hereda act, act_deterministic y la red de
    politica), sumandole una red de valor V(s) que actua de baseline:
    en vez de mover la politica con el retorno R(t) directo, se la mueve
    con la ventaja A(t) = R(t) - V(s_t)
    """

    def __init__(self, observation_space, action_space, hidden_dim=20,
                 policy_lr=1e-2, value_lr=1e-2, device="cpu"):
        super().__init__(observation_space, action_space, hidden_dim, policy_lr, device)
        input_dim = get_input_dim(observation_space)
        self.value_net = ValueNetwork(input_dim, hidden_dim).to(device)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=value_lr)

    def value(self, obs):
        """V(s): forward de la red de valor para un estado. Devuelve un escalar
        con grafo de autograd (necesario para entrenar la red de valor)"""
        state = obs_to_tensor(obs, self.observation_space, self.device)
        return self.value_net(state).squeeze(-1)

    def save(self, path):
        torch.save({"policy": self.policy.state_dict(), "value": self.value_net.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value_net.load_state_dict(checkpoint["value"])


def collect_batch(env, agent, batch_size, gamma):
    """Igual que en la Parte 2 (correr batch_size trayectorias con la politica
    actual y juntar log-probs y retornos), pero ademas se guarda V(s_t) de cada
    paso calculado con la red de valor tal como esta antes de este update"""
    batch_log_probs = []
    batch_values = []
    batch_returns = []
    episode_returns = []
    episode_lengths = []

    for _ in range(batch_size):
        obs, _ = env.reset()
        log_probs, values, rewards = [], [], []
        terminated = truncated = False

        while not (terminated or truncated):
            action, log_prob = agent.act(obs)
            value = agent.value(obs)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

        batch_log_probs.extend(log_probs)
        batch_values.extend(values)
        batch_returns.extend(discounted_returns(rewards, gamma))
        episode_returns.append(sum(rewards))
        episode_lengths.append(len(rewards))

    log_probs_tensor = torch.stack(batch_log_probs)
    values_tensor = torch.stack(batch_values)
    returns_tensor = torch.tensor(batch_returns, dtype=torch.float32, device=agent.device)
    return log_probs_tensor, values_tensor, returns_tensor, episode_returns, episode_lengths


def reinforce_baseline_update(agent, log_probs, values, returns, batch_size):
    """Un paso de entrenamiento con baseline:

    1. Ventaja A(t) = R(t) - V(s_t). Aca es donde entra values.detach(): la
       ventaja tiene que ser una constante para el gradiente de la politica
       (si no se mezclaria el update de la politica con el de la red de
       valor en el mismo backward)
    2. Update de la politica igual que en la Parte 2, pero con A(t) en vez de
       R(t) (y normalizada: reduce varianza)
    3. Ajuste de la red de valor por regresion: minimiza el error cuadratico
       medio entre V(s_t) y el retorno real R(t). Esto necesita el grafo de
       autograd de values intacto

    Devuelve ademas el desvio de R(t) y el de A(t) sobre el mismo batch,
    que es cuanta dispersion le saca el baseline al estimador
    """
    advantage = returns - values.detach()
    normalized_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    policy_loss = -(log_probs * normalized_advantage).sum() / batch_size
    agent.optimizer.zero_grad()
    policy_loss.backward()
    agent.optimizer.step()

    value_loss = (values - returns).pow(2).mean()
    agent.value_optimizer.zero_grad()
    value_loss.backward()
    agent.value_optimizer.step()

    return (policy_loss.item(), value_loss.item(),
            returns.std().item(), advantage.std().item())


def train(env, agent, n_iterations, batch_size, gamma, writer=None, progress_desc="REINFORCE+baseline"):
    """Mismo loop que en la Parte 2 (un update por batch de trayectorias) pero
    ahora se loguean dos losses en vez de uno"""
    history = {"episode": [], "episode_return": [], "episode_length": [],
               "policy_loss": [], "value_loss": [],
               "return_std": [], "advantage_std": []}
    episode_counter = 0

    report_every = max(1, n_iterations // 20)
    pbar = trange(n_iterations, desc=progress_desc, mininterval=1e9)
    for it in pbar:
        log_probs, values, returns, ep_returns, ep_lengths = collect_batch(env, agent, batch_size, gamma)
        policy_loss, value_loss, return_std, advantage_std = reinforce_baseline_update(
            agent, log_probs, values, returns, batch_size)

        for ep_return, ep_length in zip(ep_returns, ep_lengths):
            history["episode"].append(episode_counter)
            history["episode_return"].append(ep_return)
            history["episode_length"].append(ep_length)
            history["policy_loss"].append(policy_loss)
            history["value_loss"].append(value_loss)
            history["return_std"].append(return_std)
            history["advantage_std"].append(advantage_std)

            if writer is not None:
                writer.add_scalar("Performance/EpisodeReturn", ep_return, episode_counter)
                writer.add_scalar("Performance/EpisodeLength", ep_length, episode_counter)
                writer.add_scalar("Loss/train", policy_loss, episode_counter)
                writer.add_scalar("Loss/value", value_loss, episode_counter)

            episode_counter += 1

        if it % report_every == 0 or it == n_iterations - 1:
            avg_return = sum(ep_returns) / len(ep_returns)
            pbar.n = it + 1
            pbar.set_postfix(avg_return=f"{avg_return:.2f}", policy_loss=f"{policy_loss:.3f}",
                              value_loss=f"{value_loss:.3f}", refresh=False)
            pbar.refresh()

    return history
