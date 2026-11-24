########################################
# PARTIE 3: APPLIQUER EQUATION BELLMAN #
########################################

import numpy as np
import src.memory as me


def td_target(
    recompense: float,
    gamma: float,
    q_table: np.ndarray,
    prochaine_etape: int,
    done: bool
) -> float:
    """Calcule la cible de différence temporelle (TD Target) pour l'équation de Bellman.

    Estime la valeur totale attendue après une action en additionnant la récompense
    immédiate et la valeur Q maximale future actualisée (si l'épisode n'est pas terminé).

    Args:
        recompense (float): La récompense immédiate reçue après l'action.
        gamma (float): Le facteur d'actualisation (discount factor, entre 0.0 et 1.0).
        q_table (np.ndarray): La matrice des valeurs Q.
        prochaine_etape (int): L'identifiant de l'état (case) atteint après l'action.
        done (bool): Indique si l'épisode est terminé (victoire ou défaite).

    Returns:
        float: La valeur cible calculée pour mettre à jour la Q-table.
    """
    if done:
        return float(recompense)

    return float(recompense + gamma * me.max_q_value(q_table, prochaine_etape))


def calculate_td_error(
    cible: float,
    q_table: np.ndarray,
    case: int,
    action: int
) -> float:
    """Calcule l'erreur de différence temporelle (TD Error) pour un couple (état, action).

    Mesure l'écart entre la nouvelle estimation cible (TD Target) et l'ancienne
    valeur enregistrée dans la Q-table avant l'action.

    Args:
        cible (float): La valeur cible calculée par td_target.
        q_table (np.ndarray): La matrice des valeurs Q.
        case (int): L'identifiant de l'état (case) de départ.
        action (int): L'indice de l'action effectuée.

    Returns:
        float: L'écart (positif ou négatif) servant à corriger la Q-table.
    """
    return float(cible - q_table[case, action])


def update_q_value(
    q_table: np.ndarray,
    case: int,
    action: int,
    recompense: float,
    proch_etape: int,
    done: bool,
    alpha: float,
    gamma: float
) -> float:
    """Met à jour la valeur Q d'un couple (état, action) selon l'équation de Bellman.

    Calcule la cible temporelle, l'erreur associée, puis ajuste la Q-table
    proportionnellement au taux d'apprentissage alpha.

    Args:
        q_table (np.ndarray): La matrice des valeurs Q (modifiée en place).
        case (int): L'identifiant de l'état (case) de départ.
        action (int): L'indice de l'action effectuée.
        recompense (float): La récompense immédiate obtenue.
        proch_etape (int): L'identifiant de l'état atteint après l'action.
        done (bool): Indique si l'épisode est terminé.
        alpha (float): Le taux d'apprentissage (learning rate, entre 0.0 et 1.0).
        gamma (float): Le facteur d'actualisation (discount factor, entre 0.0 et 1.0).

    Returns:
        float: La nouvelle valeur Q mise à jour pour ce couple (case, action).
    """
    target = td_target(recompense, gamma, q_table, proch_etape, done)
    erreur = calculate_td_error(target, q_table, case, action)

    q_table[case, action] += alpha * erreur

    return float(q_table[case, action])

