# SYNTHETIC-DATA-GENERATION-IN-AUTONOMOUS-DRIVING


## Description

This repository contains the complete Python code implementation for the dissertation titled *"Reinforcement Learning-Enhanced Framework for Adaptive Synthetic Data Generation in Autonomous Driving"* by Qinyi Liu, submitted to the University of Liverpool for the MSc degree (July 2025).

The project develops the RL-SDG framework, which integrates a Deep Q-Network (DQN) with a conditional DCGAN to generate diverse and realistic synthetic datasets for autonomous driving. It includes baseline models (DataGAN and CycleGAN) for comparison, trained on KITTI and DrivingStereo datasets. Key features:
- Dynamic optimization of scenario parameters (e.g., lighting, vehicle density) using RL.
- Quantitative evaluation via metrics like FID, diversity score, and YOLOv8 F1-score.
- Addresses data scarcity and domain gaps in autonomous driving.

For full details, refer to the dissertation abstract and chapters (e.g., Chapter 4 for implementation).

## Repository Structure

- **rl_sdg.py**: Core script for the RL-SDG framework (DQN + conditional DCGAN).
- **datagan.py**: Implementation of DataGAN baseline (DCGAN for synthetic image generation).
- **cyclegan.py**: Implementation of CycleGAN baseline (unpaired image-to-image translation).
- **utils.py**: Utility functions for data loading, metrics calculation (e.g., FID, edge case frequency), and visualization.
- **train.py**: Scripts for training models (50 epochs) and evaluation on KITTI/DrivingStereo.
- **data/**: Folder for dataset placeholders (download KITTI and DrivingStereo separately).
- **models/**: Saved model weights (if applicable).
- **requirements.txt**: List of Python dependencies.
- **README.md**: This file.

## Installation

1. Clone the repository:
git clone https://github.com/ql909/SYNTHETIC-DATA-GENERATION-IN-AUTONOMOUS-DRIVING.git
cd SYNTHETIC-DATA-GENERATION-IN-AUTONOMOUS-DRIVING


2. Install dependencies (Python 3.7+ recommended):
pip install -r requirements.txt
