from __future__ import annotations

import numpy as np
from PIL import Image

from .envs import make_task
from .evaluate import episode_seeds

EPISODES = 5
WIDTH = 320
STRIDE = 5
FRAME_MS = 40


def save_gif(agent, config, task, path):
    """Greedy episodes on the eval seeds, drawing no randomness: the clip is what was scored."""
    env = make_task(config, task, render_mode="rgb_array")
    frames, returns = [], []
    for seed in episode_seeds(config, task, EPISODES):
        state, _ = env.reset(seed=seed)
        frames.append(np.asarray(env.render()))
        steps, total = 0, 0.0
        while True:
            state, reward, terminated, truncated, _ = env.step(agent.act_batch([state])[0])
            steps += 1
            total += float(reward)
            # Every STRIDE-th frame only: five 500 step episodes are otherwise a huge gif.
            if steps % STRIDE == 0:
                frames.append(np.asarray(env.render()))
            if terminated or truncated:
                break
        returns.append(total)
    env.close()
    height = int(frames[0].shape[0] * WIDTH / frames[0].shape[1])
    images = [Image.fromarray(frame).resize((WIDTH, height)) for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=FRAME_MS, loop=0, optimize=True)
    return {"task": task, "episodes": EPISODES, "return_mean": float(np.mean(returns)), "returns": returns, "frames": len(images), "bytes": path.stat().st_size}


def save_clips(agent, config, out, tasks):
    """One clip per named task for the finished policy, so arms can be eyeballed side by side."""
    clips = [save_gif(agent, config, task, out / "videos" / f"{task}.gif") for task in dict.fromkeys(tasks)]
    print(f"[video] {len(clips)} clips in {out / 'videos'}", flush=True)
    return clips
