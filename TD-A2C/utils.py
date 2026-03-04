from pathlib import Path

import gym
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image


def save_training_curves(reward_hist: list[float], losses: list[float], output_path: Path) -> None:
    """
    Save reward and loss curves for TD-A2C training.

    Parameters
    ----------
    reward_hist : list[float]
        Episode reward history where each element is one episode's reward.
    losses : list[float]
        Episode loss history where each element is one episode's aggregated loss.
    output_path : Path
        Target image path used to store the training curve figure.

    Returns
    -------
    None
        This function saves a figure to disk and does not return data.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(reward_hist, color='tab:blue')
    axes[0].set_title('TD-A2C CartPole-v1 Reward per Episode')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(losses, color='tab:orange')
    axes[1].set_title('TD-A2C CartPole-v1 Loss per Episode')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_policy_gif(
    net: torch.nn.Module,
    device: torch.device,
    output_path: Path,
    max_steps: int,
    fps: int = 50
) -> int:
    """
    Run a greedy CartPole-v1 rollout and save the rendered frames as a GIF.

    Parameters
    ----------
    net : torch.nn.Module
        Trained actor-critic network that returns action logits and values.
    device : torch.device
        Device used for network inference during rollout.
    output_path : Path
        Target GIF path used to store rendered frames.
    max_steps : int
        Maximum number of environment steps to record for this rollout.
    fps : int, optional
        Frames per second for playback speed in the saved GIF.

    Returns
    -------
    int
        Number of frames written to the GIF file. Returns 0 if no frame is captured.
    """
    eval_env = gym.make('CartPole-v1', render_mode='rgb_array', new_step_api=True)
    eval_state, _ = eval_env.reset(return_info=True)
    frames: list[Image.Image] = []

    was_training = net.training
    net.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            frame = eval_env.render()
            if isinstance(frame, list):
                if not frame:
                    continue
                frame = frame[-1]

            frame_array = np.asarray(frame, dtype=np.uint8)
            frames.append(Image.fromarray(frame_array))

            state_tensor = torch.as_tensor(eval_state, dtype=torch.float32, device=device)
            action_logits, _ = net(state_tensor)
            action = torch.argmax(action_logits).item()

            eval_state, _, terminated, truncated, _ = eval_env.step(action)
            if terminated or truncated:
                break
    eval_env.close()

    if was_training:
        net.train()

    if not frames:
        return 0

    duration_ms = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0
    )
    return len(frames)
