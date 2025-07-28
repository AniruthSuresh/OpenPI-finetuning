import tensorflow as tf
from huggingface_hub import snapshot_download

import tensorflow as tf
import os
import numpy as np
from PIL import Image
import json
import base64


def download_libero_dataset(cache_dir="./my_libero_data"):
    """
    Downloads the Libero dataset from Hugging Face into the specified cache directory.
    """
    snapshot_download("openvla/modified_libero_rlds", repo_type="dataset", cache_dir=cache_dir)
    print(f"Dataset downloaded to: {cache_dir}")


def decode_image(image_bytes):
    return tf.image.decode_jpeg(image_bytes)  # or decode_png if needed

def save_tensor_as_image(tensor, path):
    array = tensor.numpy()
    image = Image.fromarray(array)
    image.save(path)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def parse_and_save_tfrecord(tfrecord_path, output_dir):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    ensure_dir(output_dir)
    subdirs = {
        "image": os.path.join(output_dir, "images"),
        "wrist_image": os.path.join(output_dir, "wrist_images"),
        "joint_state": os.path.join(output_dir, "joint_states"),
        "state": os.path.join(output_dir, "states"),
        "instruction": os.path.join(output_dir, "instructions"),
        "metadata": os.path.join(output_dir, "metadata"),
    }
    for path in subdirs.values():
        ensure_dir(path)
    
    reward_file = open(os.path.join(output_dir, "rewards.csv"), "w")
    reward_file.write("index,is_first,is_last,is_terminal,reward,discount\n")

    for idx, raw_record in enumerate(raw_dataset.take(10)):  # take(10) can be removed to process all
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        features = example.features.feature

        # 1. Save scalar info
        def get_scalar(key):
            return features[key].int64_list.value[0] if features[key].int64_list.value else features[key].float_list.value[0]

        is_first = get_scalar("steps/is_first")
        is_last = get_scalar("steps/is_last")
        is_terminal = get_scalar("steps/is_terminal")
        reward = get_scalar("steps/reward")
        discount = get_scalar("steps/discount")

        reward_file.write(f"{idx},{is_first},{is_last},{is_terminal},{reward},{discount}\n")

        # 2. Save instruction
        if "steps/language_instruction" in features:
            instr = features["steps/language_instruction"].bytes_list.value[0].decode()
            with open(os.path.join(subdirs["instruction"], f"{idx:05d}.txt"), "w") as f:
                f.write(instr)

        # 3. Save metadata path
        if "episode_metadata/file_path" in features:
            path = features["episode_metadata/file_path"].bytes_list.value[0].decode()
            with open(os.path.join(subdirs["metadata"], f"{idx:05d}_meta.txt"), "w") as f:
                f.write(path)

        # 4. Save images
        if "steps/observation/image" in features:
            image_bytes = features["steps/observation/image"].bytes_list.value[0]
            image = decode_image(image_bytes)
            save_tensor_as_image(image, os.path.join(subdirs["image"], f"{idx:05d}.png"))

        if "steps/observation/wrist_image" in features:
            wrist_bytes = features["steps/observation/wrist_image"].bytes_list.value[0]
            wrist_image = decode_image(wrist_bytes)
            save_tensor_as_image(wrist_image, os.path.join(subdirs["wrist_image"], f"{idx:05d}.png"))

        # 5. Save joint_state and state
        def save_float_list(key, subdir):
            if key in features:
                arr = np.array(features[key].float_list.value)
                np.save(os.path.join(subdirs[subdir], f"{idx:05d}.npy"), arr)

        save_float_list("steps/observation/state", "state")
        save_float_list("steps/observation/joint_state", "joint_state")

    reward_file.close()
    print(f"\n✅ Finished saving data to: {output_dir}")


# Step 1: Download the dataset
#download_libero_dataset()

# Step 2: Parse and inspect one TFRecord
parse_and_save_tfrecord("/home/aniruth/Desktop/RRC/OpenPI-finetuning/src/openpi/utils/my_libero_data/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551/libero_10_no_noops/1.0.0/liber_o10-train.tfrecord-00001-of-00032" ,
                        "/home/aniruth/Desktop/RRC/OpenPI-finetuning/src/openpi/utils/single_libero_raw")