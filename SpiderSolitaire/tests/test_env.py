
import numpy as np
import numpy.typing as npt
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend for windowed plotting
from matplotlib import pyplot as plt
import sys
from pathlib import Path

# Ensure repository root (RL) is on sys.path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from SpiderSolitaire.env.spider_env import SpiderEnv

def evaluate_random_agent(env: SpiderEnv, n_episodes: int = 1000, n_bins: int = 20) -> tuple[tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]], 
                                                                                             tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]], 
                                                                                             tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]]:
    """Evaluate a random agent on the given environment. Returns binned distribution of episode rewards, completed sequences and number of tableau facedown cards."""
    rewards, seqlens, facedown_cards = [], [], []
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        terminated = False 

        while not terminated:
            action = env.action_space.sample() if not env.is_masked_actions() else env.sample_valid_action()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        seqlens.append(info['n_complete_sequences'])
        facedown_cards.append(info['n_facedown_tableau_cards'])

    rew_values, rew_bins = np.histogram(rewards, bins=n_bins, range=(min(rewards), max(rewards)))
    seq_values, seq_bins = np.histogram(seqlens, bins=n_bins, range=(min(seqlens), max(seqlens)))
    facedown_values, facedown_bins = np.histogram(facedown_cards, bins=n_bins, range=(min(facedown_cards), max(facedown_cards)))
    return (rew_values, rew_bins), (seq_values, seq_bins), (facedown_values, facedown_bins)

def plot_histogram(data: tuple[tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]], 
                               tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]], 
                               tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]], 
                   title: str="Agent performance distribution"):
    """Plot histogram from binned data."""
    rew_values, rew_bins = data[0]
    seq_values, seq_bins = data[1]
    facedown_values, facedown_bins = data[2]
    
    fig, ax = plt.subplots(1, 3, figsize=(18,5))
    ax[0].set_title('Episode rewards distribution')
    ax[1].set_title('Facedown tableau cards distribution')
    ax[2].set_title('Completed sequences distribution')

    ax[0].hist(rew_bins[:-1], rew_bins, weights=rew_values)
    ax[1].hist(facedown_bins[:-1], facedown_bins, weights=facedown_values)
    ax[2].hist(seq_bins[:-1], seq_bins, weights=seq_values)

    fig.suptitle(title)

    plt.grid()
    plt.show()

if __name__ == "__main__":
    print("Evaluating random agent performance...")
    rand_agent_metrics_distrs = evaluate_random_agent(
        SpiderEnv(
            n_suits=1,
            n_actions_limit=100,
            rewards_policy={
                "discover_card": 4,
                "free pile": 16,
                "extend sequence": 1,
                "deal cards": -0.5,
            },
            mask_legal_actions=True,
        ),
        n_episodes=1000,
        n_bins=20,
    )

    print("Plotting results...")
    plot_histogram(
        rand_agent_metrics_distrs,
        title="Random agent performance distribution over 1000 episodes (1 suit, 100 actions limit)",
    )