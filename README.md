# Mini-Transformer-Assignment
# Project Overview
This project implements a mini Transformer encoder from scratch using PyTorch to solve a synthetic sequence classification task. The objective is to understand the core mechanisms of the Transformer architecture, including multi-head attention, positional encoding, and encoder blocks

# Task Description
The model is trained to predict a binary label based on a token sequence
*  **Rule**: Predict whether the first non-padding token appears again in the second half of the sequence
*  **Vocabulary**: PAD=0, A=1, B=2, C=3, D=4
*  **Sequence Length**: Variable lengths between 6 and 20, padded to a fixed length of 20.

# Project Structure
The repository is organized as follows:
* **model.py**: Contains the core Transformer components (Positional Encoding, Multi-Head Attention, Transformer Block, and Mini Transformer classes) implemented from scratch.
* **data.py**: Handles dataset loading and preprocessing from CSV files.
* **train.py**: Implements the training loop and validation logic.
* **benchmark.py**: The main script to train and evaluate multiple model variants for comparison.
* **utils.py**: Helper functions for testing accuracy and counting model parameters.
* **README.md**: Project documentation.

# Model Variants
For the benchmark, we compare the following settings:
* **Model A**: Standard Transformer with Positional Encoding and 4 Attention Heads.
* **Model B**: Standard Transformer with Positional Encoding and 1 Attention Head.
* **Model C**: Transformer without Positional Encoding.

# How to Run
* 1. Ensure you have the dataset files (train.csv, validation.csv, test.csv) in the project directory.
* 2. Install the required libraries: pip install torch pandas.
* 3. Execute the benchmark:

# Requirements
* **PyTorch**
* **Pandas**
* **Python 3.x**
