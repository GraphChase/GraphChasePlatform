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