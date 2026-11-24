# Q-Learning — FrozenLake

## Présentation

Ce projet consiste à implémenter un algorithme de **Q-Learning**, une méthode d'apprentissage par renforcement (*Reinforcement Learning*).

L'objectif est d'entraîner un agent à résoudre l'environnement FrozenLake de Gymnasium en apprenant progressivement quelle action effectuer dans chaque état.

## Concepts abordés

- Q-Table : représentation des valeurs associées aux couples état-action.
- Epsilon-Greedy : stratégie d'exploration et d'exploitation.
- Équation de Bellman : calcul de la valeur cible.
- Temporal Difference Error : estimation de l'erreur d'apprentissage.
- Learning Rate (α) : vitesse d'apprentissage.
- Discount Factor (γ) : importance des récompenses futures.

## Architecture du projet

```text
q-learning/
├── main.py
├── requirements.txt
├── .gitignore
└── src/
    ├── __init__.py
    ├── q_table.py
    ├── policy.py
    ├── learning.py
    ├── training.py
    └── evaluation.py
```

| Module | Responsabilité |
|---|---|
| `q_table.py` | Initialisation et lecture de la Q-Table |
| `policy.py` | Gestion de la prise de décision |
| `learning.py` | Calculs liés à l'équation de Bellman |
| `training.py` | Gestion de la boucle d'entraînement |
| `evaluation.py` | Extraction et évaluation de la politique |

## Installation

Cloner le dépôt :

```bash
git clone URL_DU_DEPOT
cd q-learning
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Exécution

Lancer le programme depuis la racine du projet :

```bash
python main.py
```

## Configuration de l'entraînement

Paramètres utilisés :

| Paramètre | Valeur |
|---|---:|
| Nombre d'épisodes | 2000 |
| Learning Rate (α) | 0.8 |
| Discount Factor (γ) | 0.95 |
| Epsilon initial | 1.0 |
| Epsilon minimal | 0.01 |
| Epsilon decay | 0.995 |

## Résultats

L'agent a été entraîné pendant 2 000 épisodes, puis évalué sur 100 épisodes indépendants.

| Indicateur | Résultat |
|---|---:|
| Épisodes d'entraînement | 2 000 |
| Épisodes d'évaluation | 100 |
| Réussites | 100/100 |
| Taux de réussite | **100 %** |

L'agent a appris une politique permettant de résoudre systématiquement l'environnement FrozenLake dans sa configuration actuelle (`is_slippery=False`).

## Technologies utilisées

- Python
- NumPy
- Gymnasium

## Ressources et remerciements

Ce projet a été réalisé dans le cadre de mon apprentissage du Machine Learning, avec l'aide des ressources pédagogiques proposées par [Deep-ML](https://www.deep-ml.com/).

Le site m'a permis d'approfondir les concepts fondamentaux du Machine Learning et de mettre en pratique mes connaissances à travers une implémentation personnelle du Q-Learning.