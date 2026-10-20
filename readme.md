# mdJPT: Multi-dataset Joint Pretrain Transformer for Emotion Decoder from EEG

> 本 Fork 的重新部署入口：[重启说明与验收边界](docs/RESTART.md)。
> `main` 保留原始代码，`codex/restart-baseline` 管理本项目的部署工作。
> 当前恢复原始编码器、CDA 和 ISA 的验证；回归—排序微调尚未实现。
> 以下保留作者原始说明，不要直接使用其服务器训练配置启动本机训练。

## Project Overview
This project implements the code from the paper [mdJPT: Multi-dataset Joint Pretrain Transformer for Emotion Decoder from EEG](url). It is designed for joint training of emotion classification models across multiple EEG datasets.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Run the Pipeline
```bash
bash run_pipeline.sh
```

## Introduction

- **train_multi.py**: Pretrain feature extractor model.  
- **ext_fea.py**: Extract features using the pretrained model.  
- **train_mlp_full.py**: Perform cross-subject emotion validation on extracted features by training an MLP.

## Configuration

All training and validation parameters are defined in the configuration file `cfgs_multi/config_multi.yaml`.  
If you want to modify hyperparameters, model settings, or dataset options for your experiments, please edit this YAML file accordingly.  

## Dataset Preparation
To pre-train, validate on your own EEG dataset, or reproduce results reported in the paper, please structure your data as follows:
1. Metadata Configuration
   - Create a new YAML file describing your dataset metadata under cfgs_multi/data/.
   - You can use data_example.yaml in the same directory as a reference.
2. Data Format
   - Place your dataset in a directory where each subject corresponds to a separate .mat file (n_subs subjects → n_subs files).
   - Each .mat file must contain:
     - merged_data_all_cleaned: EEG data of shape [n_channel, n_time * fs].
     - merged_n_samples_one: Array of shape [1, n_vid / n_trial], storing sample counts.
3. Custom Data Loader
   - In src/data/io_utils.py, implement a loader function for your dataset.
   - Register this function inside get_load_data_func().
   - This is also where you should provide your dataset’s labels.
4. Configuration
   - Select your dataset in config_multi.yaml to use it for training/validation.



