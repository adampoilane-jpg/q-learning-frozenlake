################################################
# PARTIE 2: GERER LA PRISE DE DECISION        #
################################################

import numpy as np 
import gymnasium as gym
import src.memory as me

def sample_random_action(action_space: gym.Space) -> int:
    """Retourne une action échantillonnée aléatoirement depuis l'espace d'actions."""
    return int(action_space.sample())


def should_explore(epsilon: float, rng: np.random.Generator) -> bool:
    """Décide si l'agent doit explorer l'environnement ou exploiter ses connaissances.

    Tire un nombre aléatoire entre 0 et 1 et le compare au seuil epsilon.

    Args:
        epsilon (float): La probabilité d'exploration (comprise entre 0.0 et 1.0).
        rng (np.random.Generator): Le générateur de nombres aléatoires NumPy.

    Returns:
        bool: True s'il faut explorer (action aléatoire), False s'il faut
            exploiter (action gloutonne).
    """
    return bool(rng.random() < epsilon)


def select_epsilon_greedy_action(
    q_table: np.ndarray,
    case: int,
    epsilon: float,
    action_space: gym.Space,
    rng: np.random.Generator
) -> int:
    """Choisit une action selon la stratégie epsilon-greedy avec départage aléatoire.

    Explore avec une probabilité epsilon en choisissant une action au hasard.
    Sinon, exploite la Q-table en choisissant aléatoirement parmi les actions
    ayant la valeur Q maximale (pour éviter de biaiser les égalités).

    Args:
        q_table (np.ndarray): La matrice des valeurs Q.
        case (int): L'identifiant de l'état (case) actuel.
        epsilon (float): La probabilité d'exploration (entre 0.0 et 1.0).
        action_space (gym.Space): L'espace d'actions de l'environnement Gymnasium.
        rng (np.random.Generator): Le générateur aléatoire NumPy.

    Returns:
        int: L'indice de l'action choisie.
    """
    if should_explore(epsilon, rng):
        return sample_random_action(action_space)

    valeur_q = q_table[case]
    max_val = me.max_q_value(q_table, case)
    meilleures_actions = np.flatnonzero(valeur_q == max_val)

    return int(rng.choice(meilleures_actions))


def decay_epsilon(epsilon: float, decay_rate: float, min_epsilon: float) -> float:
    """Diminue progressivement le taux d'exploration epsilon (décroissance exponentielle).

    Réduit epsilon à chaque épisode en le multipliant par un facteur d'atténuation,
    tout en garantissant qu'il ne descende jamais sous une valeur minimale.

    Args:
        epsilon (float): La valeur actuelle d'epsilon (entre 0.0 et 1.0).
        decay_rate (float): Le facteur de réduction (ex: 0.995).
        min_epsilon (float): Le seuil plancher d'exploration à conserver (ex: 0.01).

    Returns:
        float: La nouvelle valeur d'epsilon mise à jour.
    """
    return float(max(min_epsilon, epsilon * decay_rate))
