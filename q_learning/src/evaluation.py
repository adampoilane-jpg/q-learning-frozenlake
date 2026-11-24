#####################################################
# PARTIE 5: EXTRAIRE ET EVALUER LA STRATEGIE FINAL #
#####################################################


import numpy as np
import gymnasium as gym


def extract_greedy_policy(q_table: np.ndarray) -> np.ndarray:
    """Extrait la politique gloutonne (optimale) déterministe depuis la Q-table.

    Pour chaque état (ligne), sélectionne l'indice de l'action ayant la valeur Q
    la plus élevée.

    Args:
        q_table (np.ndarray): La matrice entraînée des valeurs Q de forme
            (num_states, num_actions).

    Returns:
        np.ndarray: Un tableau 1D d'entiers (int64) de taille (num_states,)
            associant à chaque état sa meilleure action.
    """
    return np.argmax(q_table, axis=1).astype(np.int64)


def run_greedy_episode(
    env: gym.Env,
    policy: np.ndarray,
    seed: int | None = None,
    max_steps: int = 200
) -> bool:
    """Exécute un épisode en suivant strictement une politique déterministe et vérifie le succès.

    Réinitialise l'environnement, applique à chaque étape l'action dictée par la
    politique sans aucune exploration, et vérifie si l'objectif a été atteint.

    Args:
        env (gym.Env): L'environnement Gymnasium.
        policy (np.ndarray): Tableau 1D associant chaque état à son action optimale.
        seed (int | None, optional): Graine aléatoire pour la réinitialisation. Par défaut None.
        max_steps (int, optional): Nombre maximal de pas autorisés. Par défaut 200.

    Returns:
        bool: True si l'agent a atteint l'objectif (épisode terminé avec récompense positive),
            False sinon (défaite ou limite de pas atteinte).
    """
    state, _ = env.reset(seed=seed)

    for _ in range(max_steps):
        action = int(policy[int(state)])

        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state

        if terminated or truncated:
            return bool(terminated and float(reward) > 0)

    return False

def evaluate_success_rate(
    env: gym.Env,
    policy: np.ndarray,
    num_episodes: int,
    seed: int | None = 0,
    max_steps: int = 200
) -> float:
    """Évalue une politique déterministe sur plusieurs épisodes et calcule son taux de réussite.

    Exécute `num_episodes` parties en mode glouton (sans exploration) avec des graines
    aléatoires reproductibles et retourne la proportion d'épisodes réussis.

    Args:
        env (gym.Env): L'environnement Gymnasium.
        policy (np.ndarray): Tableau 1D associant chaque état à son action optimale.
        num_episodes (int): Nombre d'épisodes d'évaluation à exécuter.
        seed (int | None, optional): Graine de base pour la reproductibilité (incrémentée
            à chaque épisode). Si None, aucune graine n'est fixée. Par défaut 0.
        max_steps (int, optional): Nombre maximal de pas autorisés par épisode. Par défaut 200.

    Returns:
        float: Proportion d'épisodes ayant atteint l'objectif (entre 0.0 et 1.0).
    """
    if num_episodes <= 0:
        return 0.0

    successes = 0

    for ep in range(num_episodes):
        ep_seed = None if seed is None else seed + ep

        if run_greedy_episode(
            env, policy, seed=ep_seed, max_steps=max_steps
        ):
            successes += 1

    return float(successes / num_episodes)
