"""Evaluate an OpenPi policy (pi0 / pi0.5 / any VLA served by openpi) on the
bimanual UR7e sim, over the network.

The policy runs as a REMOTE websocket server (it holds the model + GPU); this
script is the CLIENT. Each control step we send {images, joint state, prompt}
and receive an action chunk, which we replay open-loop and replan periodically
-- the standard OpenPi eval pattern (see openpi examples/libero/main.py).

Start a server first, e.g. (in the openpi repo):
    uv run scripts/serve_policy.py --env=DROID    # or your own config

Install the lightweight client into THIS project's venv:
    uv pip install -e /path/to/openpi/packages/openpi-client

Then:
    uv run python openpi_eval.py --host 127.0.0.1 --port 8000 \
        --prompt "pick up the block" --episodes 5
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np

from env import SimBimanualUR7eEnv


def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize+pad to size x size, uint8. Uses openpi's image_tools when present
    (matches server-side preprocessing); falls back to cv2 otherwise."""
    try:
        from openpi_client import image_tools
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))
    except ImportError:
        import cv2
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.uint8)


def run(host: str, port: int, prompt: str, episodes: int,
        max_steps: int, replan_steps: int, resize: int, view: bool) -> None:
    # Imported here so the env stays usable without openpi installed.
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    print(f"connected to openpi policy server at {host}:{port}")

    # OpenPi policies emit raw joint targets, so use the un-normalized action space.
    env = SimBimanualUR7eEnv(normalized_actions=False, max_episode_steps=max_steps,
                          prompt=prompt, show_viewer=view)

    for ep in range(episodes):
        obs, _ = env.reset()
        action_plan: deque = deque()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            if not action_plan:
                element = env.get_openpi_observation(prompt)
                # Resize images to what the server expects.
                element["observation/image"] = _resize_image(element["observation/image"], resize)
                if len(element) > 2:
                    element["observation/wrist_image/left"] = _resize_image(
                        element["observation/wrist_image/left"], resize)
                    element["observation/wrist_image/right"] = _resize_image(
                        element["observation/wrist_image/right"], resize)
                action_chunk = client.infer(element)["actions"]
                action_plan.extend(action_chunk[:replan_steps])

            action = action_plan.popleft()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"episode {ep}: steps={steps} reward={ep_reward:.3f} "
              f"success={info.get('success')}")

    env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--prompt", default="")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--replan-steps", type=int, default=5,
                   help="actions replayed open-loop per inference before replanning")
    p.add_argument("--resize", type=int, default=224, help="image size sent to server")
    p.add_argument("--view", action="store_true",
                   help="open a live MuJoCo viewer window during eval")
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
