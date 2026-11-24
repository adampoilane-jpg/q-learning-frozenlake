#########################################
# PARTIE FINAL: LANCEMENT DU PROGRAMME #
#########################################

import gymnasium as gym
import src.training as trng
import src.evaluation as eln
import time

if __name__ == "__main__":

    # 1. Créer l'environnement (is_slippery=False pour que la glace ne glisse pas au début)
    env = gym.make("FrozenLake-v1", is_slippery=False)

    # 2. Entraîner ton agent avec ta fonction
    print("Entraînement en cours...")

    q_table, returns = trng.train_q_learning(
        env,
        num_episodes=2000,
        alpha=0.8,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        seed=42
    )

    # 3. Extraire la meilleure stratégie et mesurer le taux de réussite
    policy = eln.extract_greedy_policy(q_table)

    success_rate = eln.evaluate_success_rate(
        env, policy, num_episodes=100, seed=42
    )

    print(f"Taux de réussite après entraînement : {success_rate * 100:.1f} %")
    print("Meilleure action par case (0=Gauche, 1=Bas, 2=Droite, 3=Haut) :")
    print(policy.reshape(4, 4))

    env.close()

    # 4. Voir le lutin jouer en vrai dans une fenêtre graphique 2D !
    env_visu = gym.make(
        "FrozenLake-v1",
        is_slippery=False,
        render_mode="human"
    )

    state, _ = env_visu.reset(seed=42)

    print("\n--- Démonstration 2D en direct ---")

    for _ in range(20):
        action = int(policy[int(state)])

        state, reward, terminated, truncated, _ = env_visu.step(action)

        time.sleep(0.4)  # Pause de 0.4s entre chaque pas pour bien voir le mouvement

        if terminated or truncated:
            print("Victoire !" if float(reward) > 0 else "Tombé dans un trou !")
            time.sleep(1.5)  # Laisse la fenêtre ouverte 1.5s à la fin
            break

    env_visu.close()