# GCN-UOD

The code for our paper "Generative Cooperative Network for Unsupervised Outlier Detection".

## Environment Setup

Our project is implemented using the PyTorch framework. The following are the necessary dependencies:

- python==3.11.5
- pytorch==2.1.0
- CUDA==12.1
- numpy==1.26.0
- pandas==2.1.1
- matplotlib==3.8.0
- scipy==1.11.3
- sklearn==1.3.2

## Usage

We provide five datasets in `./data`, as referenced in our paper. Parameters for these datasets are detailed in `parameters.txt`. Below is an example illustrating the training process for our models (ACE & GCN-UOD) on the SpamBase dataset. By default, the results, including model weights, AUC scores, inference times, and training history, are stored in `./results`.

1. Train ACE to describle a preliminary boundary that separates significant outliers from normal data:
   ```
   python main.py --model ACE --data SpamBase --lr_E 1e-5 --epochs 600 --alpha 0.05
   ```

2. Copy the weights of ACE to `./weights`:
   ```
   cp path/to/weights ./weights/ACE_Spambase.pt
   ```

3. Train GCN-UOD to describle a more resonable boundary:
   ```
   python main.py --model GCN-UOD --data SpamBase --lr_E 1e-4 --lr_G 1e-4 --epochs 500 --drop_one
   ```
