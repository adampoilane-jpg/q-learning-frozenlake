###########################################
# PARTIE 4: ASSEMBLER BOUCLE ENTRAINEMENT #
###########################################


import numpy as np
import gymnasium as gym
import src.memory as me
import src.policy as pol
import src.learning as lrng



def interaction_step(
    env: gym.Env,
    q_table: np.ndarray,
    case: int,
    epsilon: float,
    alpha: float,
    gamma: float,
    rng: np.random.Generator
) -> tuple[int, float, bool]:
    """Exécute un pas complet d'interaction et d'apprentissage dans l'environnement.

    Choisit une action via epsilon-greedy, l'applique à l'environnement Gymnasium,
    met à jour la Q-table, et renvoie le nouvel état ainsi que l'état de fin d'épisode.

    Args:
        env (gym.Env): L'environnement Gymnasium.
        q_table (np.ndarray): La matrice des valeurs Q (modifiée en place).
        case (int): L'identifiant de l'état (case) actuel.
        epsilon (float): La probabilité d'exploration (entre 0.0 et 1.0).
        alpha (float): Le taux d'apprentissage (learning rate).
        gamma (float): Le facteur d'actualisation (discount factor).
        rng (np.random.Generator): Le générateur aléatoire NumPy.

    Returns:
        tuple[int, float, bool]: Un triplet (prochaine_case, recompense, done)
            indiquant le nouvel état, la récompense reçue et si l'épisode s'arrête.
    """
    action = pol.select_epsilon_greedy_action(
        q_table, case, epsilon, env.action_space, rng
    )

    proch_etape, recompense, terminer, truncated, _ = env.step(action)

    done = bool(terminer or truncated)

    lrng.update_q_value(
        q_table,
        case,
        action,
        float(recompense),
        int(proch_etape),
        bool(terminer),
        alpha,
        gamma
    )

    return (int(proch_etape), float(recompense), done)


def run_training_episode(
    env: gym.Env,
    q_table: np.ndarray,
    epsilon: float,
    alpha: float,
    gamma: float,
    rng: np.random.Generator,
    max_steps: int = 200
) -> float:
    """Exécute un épisode complet d'entraînement et met à jour la Q-table à chaque pas.

    Réinitialise l'environnement, fait interagir l'agent jusqu'à la fin de l'épisode
    ou jusqu'à atteindre la limite de pas autorisée, et cumule les récompenses.

    Args:
        env (gym.Env): L'environnement Gymnasium.
        q_table (np.ndarray): La matrice des valeurs Q (modifiée en place).
        epsilon (float): Le taux d'exploration actuel (entre 0.0 et 1.0).
        alpha (float): Le taux d'apprentissage (learning rate).
        gamma (float): Le facteur d'actualisation (discount factor).
        rng (np.random.Generator): Le générateur aléatoire NumPy.
        max_steps (int, optional): Nombre maximal de pas par épisode. Par défaut 200.

    Returns:
        float: La somme totale des récompenses obtenues durant l'épisode.
    """
    case, _ = env.reset()
    total_recompense = 0.0

    for _ in range(max_steps):
        proch_etape, recompense, done = interaction_step(
            env, q_table, int(case), epsilon, alpha, gamma, rng
        )

        total_recompense += recompense
        case = proch_etape

        if done:
            break

    return float(total_recompense)


def train_q_learning(
    env: gym.Env,
    num_episodes: int,
    alpha: float = 0.8,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.99,
    seed: int = 0,
    max_steps: int = 200
) -> tuple[np.ndarray, list[float]]:
    """Entraîne un agent par Q-learning sur un environnement Gymnasium discret.

    Initialise la Q-table et le générateur aléatoire, puis exécute une série
    d'épisodes en réduisant progressivement le taux d'exploration epsilon.

    Args:
        env (gym.Env): L'environnement Gymnasium à espaces discrets.
        num_episodes (int): Le nombre total d'épisodes d'entraînement.
        alpha (float, optional): Le taux d'apprentissage. Par défaut 0.8.
        gamma (float, optional): Le facteur d'actualisation. Par défaut 0.95.
        epsilon_start (float, optional): Le taux d'exploration initial. Par défaut 1.0.
        epsilon_min (float, optional): Le seuil minimal d'exploration. Par défaut 0.01.
        epsilon_decay (float, optional): Le facteur de décroissance d'epsilon. Par défaut 0.99.
        seed (int, optional): La graine aléatoire pour la reproductibilité. Par défaut 0.
        max_steps (int, optional): Le nombre maximal de pas par épisode. Par défaut 200.

    Returns:
        tuple[np.ndarray, list[float]]: Un couple contenant la Q-table entraînée
            et la liste des récompenses totales obtenues à chaque épisode.
    """
    rng = np.random.default_rng(seed)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    num_states = int(env.observation_space.n)
    num_actions = int(env.action_space.n)

    q_table = me.initialize_q_table(num_states, num_actions)

    epsilon = epsilon_start
    returns: list[float] = []

    for _ in range(num_episodes):
        episode_return = run_training_episode(
            env,
            q_table,
            epsilon,
            alpha,
            gamma,
            rng,
            max_steps=max_steps
        )

        returns.append(episode_return)
        epsilon = pol.decay_epsilon(epsilon, epsilon_decay, epsilon_min)

    return (q_table, returns)
