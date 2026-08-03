"""
Example script for running 10 rollouts of a DROID policy on the example environment.

Usage:

First, make sure you download the simulation assets and unpack them into the root directory of this package.

Then, in a separate terminal, launch the policy server on localhost:8000 
-- make sure to set XLA_PYTHON_CLIENT_MEM_FRACTION to avoid JAX hogging all the GPU memory.

For example, to launch a pi0-FAST-DROID policy (with joint position control), 
run the command below in a separate terminal from the openpi "karl/droid_policies" branch:

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid_jointpos --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos

Finally, run the evaluation script:

python run_eval.py --episodes 10 --headless
"""

import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from sim_evals.inference.droid_jointpos import Client as DroidJointPosClient


def main(
        episodes:int = 10,
        headless: bool = True,
        scene: int = 1,
        env_id: str = "DROID",
        task_id: str | None = None,
        external_camera: str | None = None,
        ):
    # launch omniverse app with arguments (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # All IsaacLab dependent modules should be imported after the app is launched
    import sim_evals.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg


    # Initialize the env
    env_cfg = parse_env_cfg(
        env_id,
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    instruction = None
    if env_id == "DROID-MILA":
        from sim_evals.environments.mila.tasks import MILA_TASKS, get_mila_task

        if task_id is None:
            supported_tasks = ", ".join(sorted(MILA_TASKS))
            raise ValueError(
                f"DROID-MILA requires --task-id. Supported tasks: {supported_tasks}"
            )
        task = get_mila_task(task_id)
        instruction = task.language_instruction
        env_cfg.set_task(task.task_id)
    else:
        match scene:
            case 1:
                instruction = "put the cube in the bowl"
            case 2:
                instruction = "put the can in the mug"
            case 3:
                instruction = "put banana in the bin"
            case _:
                raise ValueError(f"Scene {scene} not supported")
        env_cfg.set_scene(scene)

    env = gym.make(env_id, cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset() # need second render cycle to get correctly loaded materials
    if external_camera is None:
        external_camera = "external_cam_2" if env_id == "DROID-MILA" else "external_cam"
    client = DroidJointPosClient(
        external_camera=external_camera,
        num_envs=1,
    )


    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    ep = 0
    max_steps = env.env.max_episode_length
    with torch.no_grad():
        for ep in range(episodes):
            episode_reward = 0.0
            for _ in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                ret = client.infer(obs, [instruction])
                if not headless:
                    cv2.imshow(
                        "Right Camera",
                        cv2.cvtColor(ret["viz"][0], cv2.COLOR_RGB2BGR),
                    )
                    cv2.waitKey(1)
                video.append(ret["viz"][0])
                action = torch.tensor(ret["action"])
                obs, reward, term, trunc, _ = env.step(action)
                episode_reward += float(reward[0].detach().cpu().item())
                if term or trunc:
                    break

            client.reset()
            mediapy.write_video(
                video_dir / f"episode_{ep}.mp4",
                video,
                fps=15,
            )
            video = []

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)
