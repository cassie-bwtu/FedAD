# FedAD

**FedAD: Adaptive Block-Dropping Approach for Federated Learning on Resource-Constrained Devices**

## Introduction

FedAD is an adaptive block-dropping approach designed for federated learning environments with heterogeneous, limited, and time-varying computational resources. It dynamically adjusts the size of local training models in real-time by dropping neural network blocks based on each device's varying resource availability.

### Key Features

- **Adaptive Block Dropping**: Dynamically drops neural network blocks based on real-time resource availability
- **Policy Network**: Uses a policy network to determine the drop probability for each block depending on data inputs
- **Resource-Aware**: Each device independently generates drop policies that adapt to input data and available resources
- **High Performance**: Converges faster and outperforms state-of-the-art methods in resource-constrained settings
- **Flexible & Robust**: Strong adaptability to real-time resource fluctuations and heterogeneous resource environments

## Project Structure

```
FedAD/
├── flcore/
│   ├── clients/                
│   │   ├── clientbase.py        
│   │   ├── client_feddad.py    
│   │   ├── client_fedavg.py     
│   │   ├── client_drop.py       
│   │   └── client_freeze.py     
│   ├── servers/                 
│   │   ├── serverbase.py        
│   │   ├── server_fedad.py      
│   │   ├── server_fedavg.py     
│   │   ├── server_drop.py       
│   │   └── server_freeze.py     
│   ├── optimizers/             
│   │   └── fedoptimizer.py
│   └── trainmodel/              
│       ├── models.py            
│       ├── policy_network.py    
│       ├── regnet.py           
│       ├── regnets.py           
│       ├── resnet.py            
│       ├── densenet.py         
│       ├── efficientnet.py     
│       ├── mobilenet_v2.py     
│       └── reglayers.py         
├── utils/                       
│   ├── data_utils.py            
│   ├── gumbel_softmax.py        
│   └── result_utils.py          
└── main.py                      # Main entry point
```

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy

## Installation

```bash
git clone https://github.com/cassie-bwtu/FedAD.git
cd FedAD
```

## Usage

### Basic Command

```bash
python main.py -algo=Fedad -m=<model> -data=<dataset> -mn=<model> -nb=<Number_of_classes> -mf=<FLOPs_of_the_global_model> -basicf=<FLOPs_of_the_basic_module> -maxf=<Maximum_device_FLOPs> -nbs=<Number_of_dropable_blocks> [options]
```

### Example: Running FedAD with RegNet on CIFAR-100

```bash
python main.py -algo=Fedad -m=regnet -data=Cifar100 -datainfo=noniid_dir -mn=regnet \
    -lr=0.1 -pnlr=0.5 -nb=100 -mf=122.90 -basicf=42.68 -maxf=38.71 -nbs=9
```

### Arguments

| Argument | Description |
|----------|-------------|
| `-algo` | Algorithm to use (e.g., `Fedad`, `Fedavg`) |
| `-m` | Model architecture (e.g., `regnet`, `resnet34`) |
| `-mn` | Model name |
| `-data` | Dataset (e.g., `Cifar100`, `emnist`) |
| `-datainfo` | Data distribution type (e.g., `noniid_dir`) |
| `-lr` | Learning rate |
| `-pnlr` | Policy network learning rate |
| `-nb` | Number of classes |
| `-mf` | FLOPs of the global model |
| `-basicf` | FLOPs of the basic module in global model |
| `-maxf` | Maximum device FLOPs |
| `-nbs` | Number of dropable blocks |

### Experiment Configurations

The following table provides the recommended parameter settings for different dataset and model combinations:

| Dataset/Model | `-nb` | `-mf` | `-basicf` | `-nbs` | Hardware | Device Computing Power (TFLOPS) | Round Time T (s) |
|---------------|-------|-------|-----------|--------|----------|---------------------------------|------------------|
| CIFAR-100 / ResNet18 | 100 | 74.48 | 5.75 | 8 | NVIDIA A40 | {37.13, 22.28, 11.14} | 2.15 |
| EuroSAT-MS / ResNet34 | 62 | 139.78 | 2.20 | 16 | NVIDIA A40 | {37.13, 22.28, 11.14} | 54.75 |
| Tiny-ImageNet / ResNet34 | 200 | 600.22 | 23.21 | 16 | NVIDIA A40 | {37.13, 22.28, 11.14} | 31.78 |
| EuroSAT-MS / ResNet50 | 62 | 157.18 | 15.88 | 16 | NVIDIA RTX A6000 | {38.71, 23.23, 11.61} | 59.51 |
| CIFAR-100 / RegNet | 100 | 122.90 | 42.68 | 9 | NVIDIA RTX A6000 | {38.71, 23.23, 11.61} | 3.46 |
| Tiny-ImageNet / MobileNetV2 | 200 | 52.6 | 27.87 | 10 | NVIDIA RTX A6000 | {38.71, 23.23, 11.61} | 2.69 |

## Supported Models

- RegNet
- ResNet18, ResNet34, ResNet50
- MobileNetV2
- ViT-Small
- Bert-Mini

## Supported Algorithms

- **FedAD** (proposed method)
- FedAvg
- FedDrop


## Reproducibility

### Random Seeds
All results in the paper are averaged over 3 independent runs with seeds {7, 32, 42}, controlling client sampling, mini-batch shuffling, and parameter initialization.

### Learning Rates
Both the classifier learning rate (`-lr=0.1`) and the policy network learning rate (`-pnlr=0.5`) are held constant throughout training. SGD is used as the optimizer.

### Dataset Partitioning
Non-IID data partitioning follows the Dirichlet (β=0.1) protocol, implemented based on PFLlib (https://github.com/TsingZ0/PFLlib). Each client's local data is further split 9:1 into train/test subsets, and the global test set is the union of all client test subsets.

### Round Time T
The round time T is computed as the time required to train the average number of samples per client using the complete classifier on devices with sufficient computational resources, following Eq. (6) in the paper. Exact values for each experimental configuration are listed in `Experiment Configurations`.



