"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
import pathlib
import tensorflow as tf
import numpy as np

REPO_NAME = "Aniruth-11/libero_converted_to_lerobot_base_own"  # Name of the output dataset, also used for the Hugging Face Hub

RAW_DATASET_NAMES = [
    "libero_10_no_noops"
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    """
    # --- CUSTOM DATA LOADING LOGIC ---
    NOTE : This is a dummy implementation to illustrate how we can convert a custom dataset to LeRobot format.
    ###
    
    Find all files containing  .tfrecord in the provided data directory.
    data_path = pathlib.Path(data_dir)
    tfrecord_files = sorted(list(data_path.glob("*.tfrecord*")))

    if not tfrecord_files:
        print(f"[ERROR] No .tfrecord files found in directory: {data_dir}")
        return

    print(f"[INFO] Found {len(tfrecord_files)} TFRecord files to process.")

    # Loop over each found TFRecord file.
    for tfrecord_path in tfrecord_files:

        print(f"\n--- Processing file: {tfrecord_path.name} ---")
        raw_dataset = tf.data.TFRecordDataset(str(tfrecord_path))

        # Each record in the file is a complete episode.
        for raw_record in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            features = example.features.feature

            # Extract the flattened data arrays for the entire episode.
            state_flat = np.array(features["steps/observation/state"].float_list.value)
            action_flat = np.array(features["steps/action"].float_list.value)
            
            # Reshape the arrays to get per-step data => 8D state and 7D action.
            states = state_flat.reshape(-1, 8)
            actions = action_flat.reshape(-1, 7)
            num_steps = states.shape[0]

            # Extract the list of images and the single language instruction for the episode.
            image_list = features["steps/observation/image"].bytes_list.value
            wrist_image_list = features["steps/observation/wrist_image"].bytes_list.value
            instruction = features["steps/language_instruction"].bytes_list.value[0].decode('utf-8')

            # Loop through each step of the episode and add it as a frame.
            for step_idx in range(num_steps):
                dataset.add_frame(
                    {
                        "image": image_list[step_idx],
                        "wrist_image": wrist_image_list[step_idx],
                        "state": states[step_idx],
                        "actions": actions[step_idx],
                        "task": instruction,
                    }
                )
            
            # After adding all frames for an episode, save it.
            dataset.save_episode()
            print(f"  > Saved episode with {num_steps} steps.")
    """
    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
