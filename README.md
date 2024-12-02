## Introduction

GraphChase Platform is an open-source Python library for designing efficient learning algorithms for solving Urban Network Security Games(UNSGs). Specifically, GraphChase
offers a unified and flexible game environment for modeling various variants of
UNSGs, supporting the development, testing, and benchmarking of algorithms.


## Setup

The GraphChase Platform uses a Conda environment with Python 3.10+. Follow these steps to set up the environment and install the required packages.

### 1. Clone the Repository
```
git clone https://github.com/GraphChase/GraphChasePlatform.git
cd GraphChasePlatform
```



### 2. Install Dependencies

```
# Create a new conda environment
conda create -n nsg_env python=3.10

# Activate the environment
conda activate nsg_env

# Install PyTorch
pip install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install DGL
conda install -c dglteam/label/th21_cu121 dgl

# Install Other Requirements
pip install -r requirements.txt
```

## Example Usage

Here's an example of how to set up and run the NSGZero algorithm:

1. **Customize the Graph**: Open `graph/graph_files/custom_graph.py` and set the pursuer, evader, and exit positions along with other parameters.

2. **Adjust Algorithm Parameters**: Open the corresponding configuration file in the `configs` directory, for example, `nsgzero_configs.py`, and adjust the parameters as needed.

3. **Run the Algorithm**: Execute the script to run the NSGZero algorithm.

    ```bash
    python scripts/run_nsgzero_solver.py
    ```

The running process for other algorithms follows the same procedure as mentioned above.

## Evaluation Methods

### Overview

We provide three methods to evaluate the pursuer strategy:

1. Worst Case Utility Testing
2. Strategy Robustness Testing
3. Exploitability Testing

### 1. Worst Case Utility Testing

This method evaluates the pursuer strategy's performance in worst-case scenarios. The implementation is already available in the current training code.

### 2. Strategy Robustness Testing

This method tests the pursuer strategy's ability to handle different evader strategies by adjusting the maximum time horizon. A longer time horizon allows the evader more path choices, enabling comprehensive testing of the pursuer strategy against various evader behaviors.

#### Usage

```bash
# For NSG-Zero model
python evaluate_worst_case_utility.py nsgzero custom_horizon model_path

# For NSG-NFSP model  
python evaluate_worst_case_utility.py nsgnfsp custom_horizon model_path

# For Pretrain-PSRO model
python evaluate_worst_case_utility.py pretrainpsro custom_horizon model_path --load_node_embedding_model embedding_model_path
```

### 3. Exploitability Testing
This method assesses the pursuer strategy's vulnerability to exploitation by training an evader strategy using reinforcement learning while keeping the pursuer strategy fixed.

#### Usage

```bash
# For NSG-Zero model
python evaluate_exploit.py nsgzero custom_horizon model_path

# For NSG-NFSP model
python evaluate_exploit.py nsgnfsp custom_horizon model_path

# For Pretrain-PSRO model
python evaluate_exploit.py pretrainpsro custom_horizon model_path --load_node_embedding_model embedding_model_path
```

### Parameters

- custom_horizon: The maximum time horizon for the evaluation
- model_path: Path to the trained pursuer strategy model
- embedding_model_path: Path to the node embedding model (required for Pretrain-PSRO only)
