# gym-av-aloha
Gym environment for AV-ALOHA simulation experiments.

# AV ALOHA Simulation Datasets

| Dataset | Eye Data | Episodes | Visualization |
|---------|----------|----------|--------------|
| [AV ALOHA Sim Peg Insertion](https://huggingface.co/datasets/iantc104/av_aloha_sim_peg_insertion) | ✅ | 100 | [View](https://huggingface.co/spaces/iantc104/av_aloha_visualize_dataset?dataset=iantc104%2Fav_aloha_sim_peg_insertion&episode=0) |
| [AV ALOHA Sim Cube Transfer](https://huggingface.co/datasets/iantc104/av_aloha_sim_cube_transfer) | ✅ | 200 | [View](https://huggingface.co/spaces/iantc104/av_aloha_visualize_dataset?dataset=iantc104%2Fav_aloha_sim_cube_transfer&episode=0) |
| [AV ALOHA Sim Thread Needle](https://huggingface.co/datasets/iantc104/av_aloha_sim_thread_needle) | ✅ | 200 | [View](https://huggingface.co/spaces/iantc104/av_aloha_visualize_dataset?dataset=iantc104%2Fav_aloha_sim_thread_needle&episode=0) |
| [AV ALOHA Sim Pour Test Tube](https://huggingface.co/datasets/iantc104/av_aloha_sim_pour_test_tube) | ✅ | 100 | [View](https://huggingface.co/spaces/iantc104/av_aloha_visualize_dataset?dataset=iantc104%2Fav_aloha_sim_pour_test_tube&episode=0) |
| [AV ALOHA Sim Hook Package](https://huggingface.co/datasets/iantc104/av_aloha_sim_hook_package) | ✅ | 100 | [View](https://huggingface.co/spaces/iantc104/av_aloha_visualize_dataset?dataset=iantc104%2Fav_aloha_sim_hook_package&episode=0) |
| [AV ALOHA Sim Slot Insertion](https://huggingface.co/datasets/iantc104/av_aloha_sim_slot_insertion) | ✅ | 100 | [View](https://huggingface.co/spaces/iantc104/av_aloha_visualize_dataset?dataset=iantc104%2Fav_aloha_sim_slot_insertion&episode=0) |

# Data Collection

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/Soltanilara/gym_av_aloha.git 
pip install -e ./gym_av_aloha
```

Install additional dependencies:

```bash
pip install asyncio numba google-cloud-firestore
pip install lerobot==0.4.4
pip install git+https://github.com/deviamar/aiortc.git@9a968df30b9f808ace1f2d8e53600e8b312e0874
conda install -c conda-forge "ffmpeg=7.*"   # required by torchcodec
```

`lerobot >=0.3` writes datasets in the **v3.0** format, which is what the
current Hugging Face dataset viewer expects. It requires `av >=15`, while the
original `ian-chuang/aiortc` fork (which adds the RTP metadata channel used to
align headset gaze samples to video frames) pins `av <15`. That patch is a
single commit, so it has been rebased onto aiortc 1.15.0 — which allows
`av <18` — letting one environment serve both recording and dataset work. The
rebased fork lives at
[deviamar/aiortc@`rtp-metadata-av15`](https://github.com/deviamar/aiortc/tree/rtp-metadata-av15).

`torchcodec` decodes the dataset videos and needs ffmpeg 4–8 shared libraries;
the system ffmpeg on Ubuntu 20.04 is too old, hence the conda-forge install.

## Configuration

Add the following files to `gym_av_aloha/vr/`:

* `serviceAccountKey.json` (for Firebase authentication)
* `signalingSettings.json` (for WebRTC signaling configuration)

## Upgrading older datasets to v3.0

Datasets recorded before the lerobot upgrade are in the v2.1 format and will not
load in the current dataset viewer. Convert them in place with:

```bash
# the converter reads the old layout from a `v2.1` branch/tag, so create one first
python -c "from huggingface_hub import HfApi; HfApi().create_branch('<user>/<dataset>', branch='v2.1', repo_type='dataset')"
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id <user>/<dataset> --push-to-hub true
```

Note that `--push-to-hub` is parsed as `value.lower() == "true"`, so `--push-to-hub 1`
silently means *false* and converts locally without uploading.

## Available Environments

* `peg-insertion-v1`
* `cube-transfer-v1`
* `color-cubes-v1`
* `thread-needle-v1`
* `hook-package-v1`
* `pour-test-tube-v1`
* `slot-insertion-v1`

## Example: Recording Simulation Episodes

Navigate to the `scripts/` directory:

```bash
cd gym_av_aloha/scripts
```

Run the recording script:

```bash
python record_sim_episodes.py \
    --env_name hook-package-v1 \
    --num-episodes 100 \
    --repo-id iantc104/av_aloha_sim_hook_package \
    --root outputs \
    --task "hook package"
```