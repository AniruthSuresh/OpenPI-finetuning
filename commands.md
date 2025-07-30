# OpenPI Fine-tuning with Custom Libero Dataset

This guide provides all the commands needed to fine-tune OpenPI using a modified Libero dataset. 

[TODO]: Replace Libero data with your own xArm7 dataset . 

## Step 0: Clone the OpenPI Repository

```bash
git clone --recurse-submodules git@github.com:AniruthSuresh/OpenPI-finetuning.git

git submodule update --init --recursive
```

## Step 1: Setup Python Environment Using uv

We use uv to manage Python dependencies. Avoid using conda as it can cause issues with Hugging Face integrations in the later steps (see this [issue](https://github.com/huggingface/lerobot/issues/255)).

Once you have uv installed, run the following to set up your environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

⚠️ If You Encounter MuJoCo Build Errors...

Follow these steps to manually install MuJoCo 2.3.7:


### Step 1.1: Download MuJoCo

From your home directory (~) , run:

```bash
wget https://github.com/google-deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz
```
### Step 1.2: Extract to a Hidden Directory

```bash
# Create a hidden MuJoCo directory
mkdir ~/.mujoco

# Extract the downloaded archive
tar -xf mujoco-2.3.7-linux-x86_64.tar.gz -C ~/.mujoco
```

### Step 1.3: Add Environment Variables

Open your shell configuration file (assuming Bash):

```bash
nano ~/.bashrc
```
Add the following lines at the end of the file:

```bash
export MUJOCO_PATH="$HOME/.mujoco/mujoco-2.3.7"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco-2.3.7/bin"
```

Save ,  exit and do `source ~/.bashrc`

### Step 1.4: Re-run Installation

Navigate back to your project directory and re-run the installation:

```bash
cd OpenPI-finetuning


source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```



## Step 2: Install Required Dependencies

Install core libraries:

```bash
uv pip install tensorflow tensorflow_datasets
```

## Step 3: Generate Raw Dataset from Modified Libero Data

Replace the path with your dataset snapshot directory.

```bash
uv run /home/aniruth/Desktop/RRC/OpenPI-finetuning/examples/libero/convert_libero_data_to_lerobot.py \
  --data_dir /home/aniruth/Desktop/RRC/OpenPI-finetuning/src/openpi/utils/modified_libero_rlds \
  --push_to_hub
```


This script converts the Libero data to the LeRobot format and pushes it to the Hugging Face Hub if --push_to_hub is provided.

## Step 4: Modify the Main Config

Open the main configuration file and add your custom config around line 640.

Path to config:
```bash
/home/aniruth/Desktop/RRC/OpenPI-finetuning/scripts/compute_norm_stats.py
```

Then compute normalization statistics using:
```bash
uv run /home/aniruth/Desktop/RRC/OpenPI-finetuning/scripts/compute_norm_stats.py --config-name pi0_fast_libero_low_mem_finetune_mine
```

## Step 5: Start Training

Ensure enough GPU memory is allocated by setting:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

Run the training with:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero_low_mem_finetune_mine
```
