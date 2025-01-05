# FLaRe: Achieving Masterful and Adaptive Robot Policies with Large-Scale Reinforcement Learning Fine-Tuning

[![FLaRe](https://img.shields.io/badge/FLaRe-website-ff69b4.svg)](https://robot-flare.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2409.16578-b31b1b.svg)](https://arxiv.org/abs/2409.16578)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the code and data for the paper "FLaRe: Achieving Masterful and Adaptive Robot Policies with Large-Scale Reinforcement Learning Fine-Tuning"
## 🐍 Setting up the Python environment 🐍

### 🐳 Docker 🐳 [Recommended]

Please see the [README.md](docker/README.md) in the `docker` directory for instructions on how to build and run the docker image.

or use the pre-built image from Docker Hub:

```bash
docker pull khzeng777/spoc-rl:v2
```
then:
```bash
export CODE_PATH=/path/to/this/repo
export DATA_PATH=/path/to/data
export DOCKER_IMAGE=khzeng777/spoc-rl:v2
docker run \
    --gpus all \
    --device /dev/dri \
    --mount type=bind,source=${CODE_PATH},target=/root/spoc \
    --mount type=bind,source=${DATA_PATH},target=/root/data \
    --shm-size 50G \
    -it ${DOCKER_IMAGE}:latest
```
and use the following conda environment:
```bash
conda activate spoc
```

### 🛠 Local installation 🛠 [Not recommended]

```bash
pip install -r requirements.txt
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+966bd7758586e05d18f6181f459c0e90ba318bec
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+864913fpt2.1.2cu121
```


## 📊 Data 📊

### 📥 Downloading the training data 📥

FLaRe is trained using `astar` from [SPOC](https://spoc-robot.github.io/) CHORES benchamrk. The `astar` type has the agent navigating and fetching one of fifteen possible object types. To download the training data for the `astar` type, run the following command:  

```bash
python -m scripts.download_training_data --save_dir /your/local/save/dir --types astar
```

for example
```bash
python -m scripts.download_training_data --save_dir data --types astar
```

#### 📁 Dataset format 📁

Once you run the above command, you will have a directory structure that looks like this
```
/your/local/save/dir/<astar OR all>_type
    <TASK_TYPE>
        house_id_to_sub_house_id_train.json # This file contains a mapping that's needed for train data loading
        house_id_to_sub_house_id_val.json   # This file contains a mapping that's needed for val data loading
        train
            <HOUSEID>
                hdf5_sensors.hdf5 -- containing all the sensors that are not videos
                    <EPISODE_NUMBER>
                        <SENSOR_NAME>
                raw_navigation_camera__<EPISODE_NUMBER>.mp4
                raw_manipulation_camera__<EPISODE_NUMBER>.mp4
        val
            # As with train
```


The `hdf5_sensors.hdf5` contains the necessary information to train FLaRe, including the house id, starting pose, and target object type/id.

For more information about the downloaded data, including trajectory videos and recorded sensors, please refer to [SPOC](https://spoc-robot.github.io/) documentation.

## 🏋 Training and Evaluation 🏋

In order to run training and evaluation you'll need:

1. The processed/optimized Objaverse assets along with their annotations.
2. The set of ProcTHOR-Objaverse houses you'd like to train/evaluate on.
3. For evaluation only, a trained model checkpoint.

Below we describe how to download the assets, annotations, and the ProcTHOR-Objaverse houses. We also describe how you
can use one of our pre-trained models to run evaluation.

### 💾 Downloading assets, annotations, and houses 💾

#### 📦 Downloading optimized Objaverse assets and annotations 📦

Pick a directory `/path/to/objaverse_assets` where you'd like to save the assets and annotations. Then run the following commands:

```bash
python -m objathor.dataset.download_annotations --version 2023_07_28 --path /path/to/objaverse_assets
python -m objathor.dataset.download_assets --version 2023_07_28 --path /path/to/objaverse_assets
```

These will create the directory structure:
```
/path/to/objaverse_assets
    2023_07_28
        annotations.json.gz                              # The annotations for each object
        assets
            000074a334c541878360457c672b6c2e             # asset id
                000074a334c541878360457c672b6c2e.pkl.gz
                albedo.jpg
                emission.jpg
                normal.jpg
                thor_metadata.json
            ... #  39663 more asset directories
```

#### 🏠 Downloading ProcTHOR-Objaverse houses 🏠

Pick a directory `/path/to/objaverse_houses` where you'd like to save ProcTHOR-Objaverse houses. Then run: 
```bash
python -m scripts.download_objaverse_houses --save_dir /path/to/objaverse_houses --subset val
```
to download the validation set of houses as `/path/to/objaverse_houses/val.jsonl.gz`.
You can also change `val` to `train` to download the training set of houses.

#### 🛣 Setting environment variables 🛣

Next you need to set the following environment variables:
```bash
export PYTHONPATH=/path/to/flare
export OBJAVERSE_HOUSES_DIR=/path/to/objaverse_houses
export OBJAVERSE_DATA_DIR=/path/to/objaverse_assets
```

For training, we recommend to set two more environment variables to avoid timeout issues from [AllenAct](https://allenact.org/):
```bash
export ALLENACT_DEBUG=True
export ALLENACT_DEBUG_VST_TIMEOUT=2000
```

### 🚀 Running RL finetuning 🚀
Download pretrained IL ckpt:
```bash
python scripts/download_trained_ckpt.py --ckpt_ids spoc_IL --save_dir PATH_TO_SAVE_DIR
```

```bash
python training/online/dinov2_vits_tsfm_rgb_augment_objectnav.py train --il_ckpt_path IL_CKPT_PATH --num_train_processes NUM_OF_TRAIN_PROCESSES --output_dir PATH_TO_RESULT --dataset_dir PATH_TO_DATASET
```

for example
```bash
python training/online/dinov2_vits_tsfm_rgb_augment_objectnav.py train --il_ckpt_path ckpt/spoc_IL/model.pt --num_train_processes 32 --output_dir results --dataset_dir data/astar/ObjectNavType
```

### 🚀 (Optional) Running IL training 🚀
While it would be easier to directly download our pre-trained model, it is possible to retrain the IL model from
scratch through the following command:
```bash
export LONG_ACTION_NAME=1
export PYTHONPATH=/path/to/flare
export OBJAVERSE_HOUSES_DIR=/path/to/objaverse_houses
export OBJAVERSE_DATA_DIR=/path/to/objaverse_assets
python -m training.offline.train_pl --max_samples 10000000 --eval_max_samples 100 --eval_every 400 --model_version small_3 --sliding_window 100 --per_gpu_batch 16 --lr 0.0002 --data_dir PATH_TO_DATA --dataset_version CHORES --model EarlyFusionCnnTransformer --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand --precision 16-mixed --resume_local --output_dir OUTPUT_DIR --loss action --max_epochs 400
```


### 🚀 Running evaluation with a trained model 🚀

Download trained ckpt:
```bash
python scripts/download_trained_ckpt.py --save_dir PATH_TO_SAVE_DIR --ckpt_ids TaskType
```

for example:
```bash
python scripts/download_trained_ckpt.py --save_dir ckpt --ckpt_ids pickup
```

Run evaluation:
```bash
python training/online/online_eval.py --shuffle --eval_subset minival --output_basedir DIR --test_augmentation --task_type TaskType --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand --house_set objaverse --num_workers NUM_WORKERS
```

## 📝 Cite us 📝

```bibtex
@article{
        hu2024flare,
        title={FLaRe: Achieving Masterful and Adaptive Robot Policies with Large-Scale Reinforcement Learning Fine-Tuning},
        author={Jiaheng Hu and Rose Hendrix and Ali Farhadi and Aniruddha Kembhavi and Roberto Martin-Martin and Peter Stone and Kuo-Hao Zeng and Kiana Ehsani},
        journal={arXiv},
        year={2024},
        eprint={2409.16578},
}
```
