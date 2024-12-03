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

We provide four methods to evaluate the pursuer strategy:

1. Pseudo Worst Case Utility
2. Worst Case Utility Testing
3. Strategy Robustness Testing
4. Exploitability Testing

### 1. Pseudo Worst Case Utility
This method evaluates the performance of the pursuer's strategy by first selecting an exit accroding to the best response and then choosing a path within that exit. Due to the randomness in path selection, it is referred to as pseudo worst-case utility. 

### 2. Worst Case Utility
This method evaluates the performance of the pursuer's strategy by enumerating every feasible path and using the path with the lowest reward as the worst case utility.

### 3. Strategy Robustness Testing

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

If the parameter `time_horizon` is consistent with the setting during training, the evaluation method is the second one; otherwise, the third evaluation method is used.

### 4. Exploitability Testing
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

### Equilibrium Assessment

Our platform supports calculating the NashConv metric to measure convergence. This metric is calculated as:
```math
\begin{align*}
\text{NashConv} &= \text{pursuer\_br\_value}  +  \text{evader\_br\_value}
\end{align*}
```
where `pursuer_br_value` and `evader_br_value` denote the values of their respective best response strategies. The evaluation can be performed either by computing the best response during evaluation or by using pre-trained best response strategies.

### Computing Best Response During Evaluation
To compute the best response during evaluation using different algorithms:

```bash
python evaluate_nashconv.py method pursuer_model_path evader_method --graph_id graph
```

For example, to use NSG-NFSP method and evader uses worst case utility to calculate the value of best response strategy:

```bash
python evaluate_nashconv.py nsgnfsp pursuer_model_path all_path --graph_id 0
```

If evader uses pseudo worst case utility, run
```bash
python evaluate_nashconv.py nsgnfsp pursuer_model_path exit_node --graph_id 0
```

If evader uses RL algorithm, run
```bash
python evaluate_nashconv.py nsgnfsp pursuer_model_path evader_model_path --graph_id 0
```

### Using Pre-trained Best Response
To evaluate using pre-trained best response strategies:
```bash
python evaluate_nashconv.py best_response_strategy_path pursuer_model_path evader_model_path --graph_id 0
```

### Parameters

- method: The method used to compute best response (e.g., nsgnfsp)
- pursuer_model_path: Path to the pursuer model
- evader_model_path: Path to the evader model
- best_response_strategy_path: Path to the pre-trained best response strategy
- graph_id: the graph to evaluate on