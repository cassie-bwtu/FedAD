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
    -lr=0.1 -pnlr=5 -nb=100 -mf=122.90 -basicf=42.68 -maxf=38.71 -nbs=9
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

| Dataset/Model | `-nb` | `-mf` | `-basicf` | `-nbs` |
|---------------|-------|-------|-----------|--------|
| CIFAR-100 / ResNet18 | 100 | 74.48 | 5.75 | 8 |
| EuroSAT-MS / ResNet34 | 62 | 139.78 | 2.20 | 16 |
| Tiny-ImageNet / ResNet34 | 200 | 600.22 | 23.21 | 16 |
| EuroSAT-MS / ResNet50 | 62 | 157.18 | 15.88 | 16 |
| CIFAR-100 / RegNet | 100 | 122.90 | 42.68 | 9 |
| Tiny-ImageNet / MobileNetV2 | 200 | 52.6 | 27.87 | 10 |

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

