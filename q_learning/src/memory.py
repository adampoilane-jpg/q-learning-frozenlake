#####################################################
# PARTIE 1: PREPARATION MEMOIRE + OUTILS DE LECTURE #
#####################################################

import numpy as np


def initialize_q_table(nombre_case: int, nombre_action: int) -> np.ndarray:
    """Initialise la Q-table avec des zéros avant l'entraînement de l'agent.

    Crée une matrice représentant l'espace d'états et d'actions de
    l'environnement, servant de mémoire initiale à l'algorithme de Q-learning.

    Args:
        nombre_case (int): Le nombre total d'états (cases) dans l'environnement.
        nombre_action (int): Le nombre total d'actions possibles par état.

    Returns:
        np.ndarray: Une matrice 2D de forme (nombre_case, nombre_action)
            remplie de zéros de type float64.
    """
    return np.zeros((nombre_case, nombre_action), dtype=np.float64)


def max_q_value(q_table: np.ndarray, case: int) -> float:
    """Récupère la valeur Q maximale disponible pour une case donnée.

    Sert à estimer la récompense future maximale atteignable depuis cet état
    lors de la mise à jour de la Q-table (équation de Bellman).

    Args:
        q_table (np.ndarray): La matrice des valeurs Q.
        case (int): L'identifiant de l'état (case) actuel.

    Returns:
        float: Le score Q le plus élevé parmi toutes les actions possibles sur cette case.
    """
    return float(np.max(q_table[case]))


def select_greedy_action(q_table: np.ndarray, case: int) -> int:
    """Choisit l'action gloutonne (la meilleure connue) pour une case donnée.

    Sélectionne l'indice de l'action ayant la valeur Q la plus élevée sur la
    ligne correspondant à l'état actuel de l'agent.

    Args:
        q_table (np.ndarray): La matrice des valeurs Q.
        case (int): L'identifiant de l'état (case) actuel.

    Returns:
        int: L'indice de l'action optimale à effectuer (ex: 0, 1, 2 ou 3).
    """
    return int(np.argmax(q_table[case]))