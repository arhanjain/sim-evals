import numpy as np
from openpi_client import image_tools

from .abstract_client import InferenceClient
from .test import WebsocketClientPolicy


class Client(InferenceClient):
    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int = 8,
        num_envs: int = 1,
        external_camera: str = "external_cam",
    ) -> None:
        self.open_loop_horizon = open_loop_horizon
        self.num_envs = num_envs
        self.external_camera = external_camera
        self.client = WebsocketClientPolicy(remote_host, remote_port)

        self._chunk_counters = np.zeros(num_envs, dtype=np.int32)
        self._action_chunks = None

    def reset(self, env_ids: list[int] | None = None):
        """Reset chunk state for all environments or a selected subset."""
        if env_ids is None:
            self._chunk_counters[:] = 0
            self._action_chunks = None
        else:
            self._chunk_counters[env_ids] = self.open_loop_horizon

    def infer(self, obs: dict, instruction: str | list[str]) -> dict:
        """Run batched inference and return one action and visualization per environment."""
        prompts = (
            [instruction] * self.num_envs
            if isinstance(instruction, str)
            else list(instruction)
        )
        if len(prompts) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} prompts, received {len(prompts)}"
            )
        curr_obs = self._extract_observation(obs)

        needs_new = self._chunk_counters >= self.open_loop_horizon
        if self._action_chunks is None or needs_new.any():
            request_data = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(
                    curr_obs["right_image"], 224, 224
                ),
                "observation/wrist_image_left": image_tools.resize_with_pad(
                    curr_obs["wrist_image"], 224, 224
                ),
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": prompts,
            }
            new_chunks = np.asarray(self.client.infer(request_data)["actions"])
            if new_chunks.ndim == 2 and self.num_envs == 1:
                new_chunks = new_chunks[None, ...]
            if new_chunks.ndim != 3 or new_chunks.shape[0] != self.num_envs:
                raise ValueError(
                    "Policy actions must have shape "
                    f"({self.num_envs}, chunk_len, action_dim); received {new_chunks.shape}"
                )
            if self._action_chunks is None:
                self._action_chunks = new_chunks.copy()
                self._chunk_counters[:] = 0
            else:
                update_ids = np.where(needs_new)[0]
                self._action_chunks[update_ids] = new_chunks[update_ids]
                self._chunk_counters[update_ids] = 0

        actions = self._action_chunks[
            np.arange(self.num_envs), self._chunk_counters
        ]
        self._chunk_counters += 1

        gripper = (actions[:, -1] > 0.5).astype(np.float32)
        actions = np.concatenate([actions[:, :-1], gripper[:, None]], axis=1)

        img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        viz = np.concatenate([img1, img2], axis=2)

        return {"action": actions, "viz": viz}

    def _extract_observation(self, obs_dict):
        """Extract observations while preserving the environment batch dimension."""
        policy_obs = obs_dict["policy"]
        if self.external_camera not in policy_obs:
            supported = ", ".join(sorted(policy_obs))
            raise KeyError(
                f"Policy observation does not contain external camera "
                f"'{self.external_camera}'. Available observations: {supported}"
            )
        return {
            "right_image": policy_obs[self.external_camera]
            .clone()
            .detach()
            .cpu()
            .numpy(),
            "wrist_image": policy_obs["wrist_cam"]
            .clone()
            .detach()
            .cpu()
            .numpy(),
            "joint_position": policy_obs["arm_joint_pos"]
            .clone()
            .detach()
            .cpu()
            .numpy(),
            "gripper_position": policy_obs["gripper_pos"]
            .clone()
            .detach()
            .cpu()
            .numpy(),
        }
