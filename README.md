# LiFedST
Official PyTorch implementation of "LiFedST: a Linearized Federated Split-attention Transformer for spatio-temporal forecasting"

## Setup
### Environment
PyTorch 2.0.0 and conducted on NVIDIA GeForce RTX 4090 GPU with CUDA 11.8.
```bash
conda create -n LiFedST "python=3.11"
conda activate LiFedST
bash install.sh
```

## Run
```bash
bash run.sh
```
or
```
python run.py
```

## Default Hyperparameter Configuration
| Hyperparameters               | Values |
|-------------------------------|--------|
| embedding dimension                 | 16     |
| transformer dimension               | 512    |
| num_heads                           | 8      |
| polynomial coefficient $K$          | 2      |
| learning rate $\eta$                | 0.003  |
| batch size                              | 64     |
| number of global epochs $R_g$       | 200    |
| number of local epochs $R_l$        | 2      |
| validation ratio                        | 0.2    |
| testing ratio                           | 0.2    |

## Acknowledgement
We appreciate the following GitHub repos a lot for their valuable code and efforts.
- FedGTP (https://github.com/cwt-2021/KDD2024_FedGTP)
