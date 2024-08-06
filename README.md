# Inverse Evolution Data Augmentation

## Description
This project explores the application of Inverse Evolution for data augmentation in neural operators. It includes implementations of inverse evolution data augmentation for Burgers' equation, Allen-Cahn equation and Navier-Stokes equation.

## Data Download

To get started, download the dataset from the following link:

- [Download Dataset](https://connectpolyu-my.sharepoint.com/:f:/g/personal/19099128r_connect_polyu_hk/EvmxMlaEw5FIp_qnCnvWanUBW_0CejtzSYcCQXxb8YuROg?e=ns3gSO)

Ensure that you place the downloaded dataset in the appropriate directory.

## How to Run the Code

### Prerequisites

Install the necessary packages via the following command:
  ```
  pip install -r requirements.txt
  ```

### Train the neural operator
1. Navigate to the ie-data-augmentation directory:
```
   cd /ie-data-augmentation
```
2. The configuration files are located in the "configure" directory. You can adjust the parameters by modifying the values in these files. By default, the settings are for benchmarks without data augmentation. To use     
   data augmentation, set "data_augmentation" to "true". Make sure to update the data path and model save path in the configuration files as needed.
3. To train the neural operator for a specific PDE, run the corresponding script. For example, to train the neural operator for the Allen-Cahn equation, use the following command:
```
   python train_allen_cahn.py
```
