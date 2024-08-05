# Inverse Evolution Data Augmentation

## Description
This project explores the application of Inverse Evolution for data augmentation in neural operators. It includes implementations of inverse evolution data augmentation for Burgers' equation, Allen-Cahn equation and Navier-Stokes equation.

## Data Download

To get started, you need to download the dataset. You can find the dataset at the following link:

- [Download Dataset](https://example.com/dataset-link)

Ensure that you place the downloaded dataset in the appropriate directory and revise the data path in configuration files.

## How to Run the Code

### Prerequisites

Install neccessary packages:
  ```
  pip install -r requirements.txt
  ```

### Train the neural operator

 Change the "data_augmentation" in configure file to train the neural operator with or without data augmentation and then run:
   ```
   python train_allen_cahn.py
  ```
