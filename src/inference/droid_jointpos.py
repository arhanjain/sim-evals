import numpy as np
import requests
from PIL import Image
import json_numpy
from openpi_client import image_tools, websocket_client_policy

from .abstract_client import InferenceClient


class Client(InferenceClient):
    def __init__(self,
                 remote_host: str = "localhost",
                 remote_port: int = 8000,
                 open_loop_horizon: int = 8,
                 ) -> None:
        self.open_loop_horizon = open_loop_horizon
        # self.client = websocket_client_policy.WebsocketClientPolicy(
        #     remote_host, remote_port
        # )

        json_numpy.patch()
        self.client_session = requests.Session()
        self.client_url = f"http://{remote_host}:{remote_port}"

        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        base_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        combined = np.concatenate([base_img, wrist_img], axis=1)
        return combined

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def infer(self, obs: dict, instruction: str) -> dict:
        """
        Infer the next action from the policy in a server-client setup
        """
        curr_obs = self._extract_observation(obs)
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0
            # request_data = {
            #     "observation/exterior_image_1_left": image_tools.resize_with_pad(
            #         curr_obs["right_image"], 224, 224
            #     ),
            #     "observation/wrist_image_left": image_tools.resize_with_pad(
            #         curr_obs["wrist_image"], 224, 224
            #     ),
            #     "observation/joint_position": curr_obs["joint_position"],
            #     "observation/gripper_position": curr_obs["gripper_position"],
            #     "prompt": instruction,
            # }
            # result = self.client.infer(request_data)
            # self.pred_action_chunk = result['actions']
            # print(instruction)

            request_data = {
                "observation": {
                    "video.exterior_image_1": image_tools.resize_with_pad(curr_obs["right_image"], 256, 256).reshape(1, 256, 256, 3),
                    "video.wrist_image": image_tools.resize_with_pad(curr_obs["wrist_image"], 256, 256).reshape(1, 256, 256, 3),
                    "state.joint_position": curr_obs["joint_position"].reshape(1, 7),  # 7
                    "state.gripper_position": curr_obs["gripper_position"].reshape(1, 1),  # 1
                    "annotation.language.language_instruction": [instruction],
                },
            }
            response = self.client_session.post(self.client_url + "/act", json=request_data)
            result = response.json()
            self.pred_action_chunk = np.hstack([result["action.joint_position"], result["action.gripper_position"].reshape(-1, 1)])

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        both = np.concatenate([img1, img2], axis=1)

        return {"action": action, "viz": both}

    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        # Assign images
        right_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()

        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }


if __name__ == "__main__":
    import torch
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote_host", type=str, default="localhost")
    parser.add_argument("--remote_port", type=int, default=5555)
    parser.add_argument("--open_loop_horizon", type=int, default=8)
    args = parser.parse_args()
    # args = tyro.cli(Args)
    client = Client(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        open_loop_horizon=args.open_loop_horizon
    )
    fake_obs = {
        "policy": {
            "external_cam": [torch.zeros((224, 224, 3), dtype=torch.uint8)],
            "wrist_cam": [torch.zeros((224, 224, 3), dtype=torch.uint8)],
            "arm_joint_pos": torch.zeros((7,), dtype=torch.float32),
            "gripper_pos": torch.zeros((1,), dtype=torch.float32),
        },
    }
    fake_instruction = "pick up the object"

    import time

    start = time.time()
    client.infer(fake_obs, fake_instruction)  # warm up
    num = 20
    for i in range(num):
        ret = client.infer(fake_obs, fake_instruction)
        print(ret["action"].shape)
    end = time.time()

    print(f"Average inference time: {(end - start) / num}")
