# Malaria Detection using CNNs

## Overview

This project uses Convolutional Neural Networks (CNNs) to detect malaria from thin blood smear slide images. The dataset used contains 27,558 images of segmented cells.

## Dataset

The dataset is available in the TensorFlow datasets library. You can find it [here](https://www.tensorflow.org/datasets/catalog/malaria).

## Model Architecture

The CNN model has the following layers:

1. Conv2D layer with 32 filters
2. Conv2D layer with 64 filters
3. Conv2D layer with 128 filters
4. Dense layer with 128 neurons

![Model Architecture](model.png)

## Code Structure

- **downloadData.py**: Script to load the dataset.
- **training.py**: Script to train the model.
- **Test.py**: Script to visualize predictions.
- **visualize.py**: Script to generate a visualization of the model architecture.

## Demo Results

The trained model is saved in `MalariaModel.keras` and achieves an accuracy of 94%. Below are some sample results:

- **0**: Infected cell
- **1**: Uninfected cell

![Predictions](predictions.png)

## Understanding Model Predictions

It's important to note that the presence of dots in thin blood smear slides does not necessarily indicate malaria infection. The CNN model is trained to recognize specific morphological features of the Plasmodium parasite, which causes malaria. These features include:

- Shape, size, and location of the parasite within the cell
- Specific patterns and structures associated with different life stages of the parasite

The model disregards non-specific dots that may be artifacts, debris, or other cellular structures unrelated to the malaria parasite. This ensures accurate detection based on medically relevant criteria.

## Dependencies

- `tensorflow`
- `tensorflow_datasets`
- `matplotlib` (for visualizations in `Test.py`)
- `graphviz`, `pydot` (for generating model architecture in `visualize.py`)

## Setup

- **Hardware**: NVIDIA GTX 1060 with 6 GB VRAM
- **Environment**: WSL2 was used for training due to easier TensorFlow GPU setup on Linux.
- **Training Time**: Approximately 322 seconds (5 minutes and 22 seconds)
- **Convergence**: Achieved in 4 epochs. Accuracy jumped from 67% to 87% in the 3rd epoch and from 87% to 95% (for training set) in the 4th epoch. Marginal improvements were observed for 7 additional epochs.
