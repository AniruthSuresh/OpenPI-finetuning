import os
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download

# --- Helper Functions ---

def ensure_dir(directory_path):
    """Creates a directory if it does not already exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def decode_image(image_bytes):
    """Decodes a byte string into a TensorFlow image tensor."""
    return tf.io.decode_image(image_bytes, channels=3)

def save_tensor_as_image(tensor, file_path):
    """Converts a TensorFlow tensor to a PIL Image and saves it."""
    # Convert tensor to numpy array if it's not already
    if hasattr(tensor, 'numpy'):
        tensor = tensor.numpy()
    # Create PIL image and save
    image = Image.fromarray(tensor)
    image.save(file_path)

# --- Main Parsing and Download Functions ---

def download_libero_dataset(cache_dir="./my_libero_data"):
    """
    Downloads the Libero dataset from Hugging Face into the specified cache directory.
    Returns the path to the downloaded snapshot.
    """
    return snapshot_download("openvla/modified_libero_rlds", repo_type="dataset", cache_dir=cache_dir)

def extract_data_for_finetuning(tfrecord_path, output_dir):
    """
    Parses a TFRecord file where each record is a full episode, and extracts
    the 4 key data types for fine-tuning, organizing the output into folders.

    Args:
        tfrecord_path (str): The path to the .tfrecord file.
        output_dir (str): The root directory where the extracted data will be saved.
    """
    # Create the main output directory
    ensure_dir(output_dir)
    print(f"[INFO] Starting extraction from {tfrecord_path}")
    print(f"[INFO] Output will be saved to {output_dir}")

    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # In this format, each record IS an episode. So we loop through them.
    for episode_idx, raw_record in enumerate(raw_dataset):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        # --- Create Directories for the Episode ---
        episode_dir = os.path.join(output_dir, f"episode_{episode_idx:04d}")
        ensure_dir(episode_dir)
        
        image_dir = os.path.join(episode_dir, "images")
        wrist_image_dir = os.path.join(episode_dir, "wrist_images")
        state_dir = os.path.join(episode_dir, "states")
        action_dir = os.path.join(episode_dir, "actions")
        
        ensure_dir(image_dir)
        ensure_dir(wrist_image_dir)
        ensure_dir(state_dir)
        ensure_dir(action_dir)
        
        print(f"---> Processing Episode {episode_idx}, saving to: {episode_dir}")

        # --- Extract and Reshape Time-Series Data ---
        
        # 1. ROBOT STATE: The state data is a flattened list. We reshape it.
        # The shape is (num_steps, 8) for the Panda arm.
        state_feature = features.get("steps/observation/state", None)
        if not state_feature: continue # Skip if episode has no state data
        
        state_array_flat = np.array(state_feature.float_list.value)
        # Reshape from flat array to (num_steps, state_dim)
        states = state_array_flat.reshape(-1, 8) 
        num_steps = states.shape[0]
        
        # 2. ROBOT ACTION: The action data is also flattened. We reshape it.
        # The shape is (num_steps, 7) for the Panda arm.
        action_feature = features.get("steps/action", None)
        if not action_feature: continue # Skip if episode has no action data
        
        action_array_flat = np.array(action_feature.float_list.value)
        # Reshape from flat array to (num_steps, action_dim)
        actions = action_array_flat.reshape(-1, 7)

        # 3. CAMERA VIEWS: The images are stored as a list of byte strings.
        image_feature = features.get("steps/observation/image", None)
        wrist_image_feature = features.get("steps/observation/wrist_image", None)
        
        # 4. LANGUAGE INSTRUCTION: This is stored once per episode.
        instruction_feature = features.get("steps/language_instruction", None)
        if instruction_feature:
            instruction_text = instruction_feature.bytes_list.value[0].decode('utf-8')
            instruction_path = os.path.join(episode_dir, "instruction.txt")
            with open(instruction_path, "w") as f:
                f.write(instruction_text)
                
        # --- Loop through each STEP of the episode and save the data ---
        print(f"     Episode contains {num_steps} steps.")
        for step_idx in range(num_steps):
            # Save state for this step
            np.save(os.path.join(state_dir, f"{step_idx:05d}.npy"), states[step_idx])
            
            # Save action for this step
            np.save(os.path.join(action_dir, f"{step_idx:05d}.npy"), actions[step_idx])
            
            # Save images for this step
            if image_feature and len(image_feature.bytes_list.value) > step_idx:
                image_bytes = image_feature.bytes_list.value[step_idx]
                image_tensor = decode_image(image_bytes)
                save_tensor_as_image(image_tensor, os.path.join(image_dir, f"{step_idx:05d}.png"))
            
            if wrist_image_feature and len(wrist_image_feature.bytes_list.value) > step_idx:
                wrist_image_bytes = wrist_image_feature.bytes_list.value[step_idx]
                wrist_image_tensor = decode_image(wrist_image_bytes)
                save_tensor_as_image(wrist_image_tensor, os.path.join(wrist_image_dir, f"{step_idx:05d}.png"))

    print(f"\n✅ Finished! Extracted a total of {episode_idx + 1} episodes.")


if __name__ == '__main__':
    # --- Step 1: Download the dataset ---
    print("Downloading dataset...")
    snapshot_path = download_libero_dataset(cache_dir="./my_libero_data")
    print("Download complete.")

    # --- Step 2: Define paths and parse one TFRecord file ---
    tfrecord_to_parse = os.path.join(
        snapshot_path, 
        "libero_10_no_noops/1.0.0/liber_o10-train.tfrecord-00001-of-00032"
    )
    
    output_directory = "./single_libero_raw"
    
    if os.path.exists(tfrecord_to_parse):
        extract_data_for_finetuning(tfrecord_to_parse, output_directory)
    else:
        print(f"[ERROR] TFRecord file not found at: {tfrecord_to_parse}")
