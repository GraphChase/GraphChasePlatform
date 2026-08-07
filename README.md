## Introduction

Urban Network Security Games (UNSGs) model real-world scenarios in which law
enforcement must strategically allocate limited resources to intercept criminals
escaping through urban networks. Research on UNSGs has been hindered by the lack
of a standardized experimental platform and realistic benchmarks that account for
heterogeneous travel costs.

GraphChase is an open-source platform for developing and evaluating UNSG
algorithms. It provides unified environments for diverse UNSG variants on both
unweighted and weighted road networks across a range of urban topologies, together
with learning-based baseline algorithms. By supporting realistic travel-time
heterogeneity, GraphChase offers a testbed for studying the robustness, scalability,
and sim-to-real generalization of UNSG solvers.

For more details, see our paper
[GraphChase: A Platform and Benchmark for Urban Network Security Games](https://doi.org/10.1145/3770855.3817540),
published in the *Proceedings of the 32nd ACM SIGKDD Conference on Knowledge
Discovery and Data Mining (KDD 2026)*.


## Setup

The GraphChase Platform uses a Conda environment with Python 3.10+. Follow these steps to set up the environment and install the required packages.

### 1. Clone the Repository
```
git clone https://github.com/GraphChase/GraphChasePlatform.git
cd GraphChasePlatform
```



### 2. Install Dependencies

```
conda create -n nsg_env python=3.10 -y
conda activate nsg_env

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `requirements.txt` file already contains the CUDA 12.1 package sources for `torch` and `dgl`, so no extra install commands are needed.


## Example Usage

Use `run_bash.sh` to launch all supported training pipelines in sequence.

From the repository root:

```bash
bash run_bash.sh
```

`run_bash.sh` currently runs:

1. `graphchase.scripts.run_cfrmix`
2. `graphchase.scripts.run_grasper_mappo`
3. `graphchase.scripts.run_nsgnfsp`
4. `graphchase.scripts.run_nsgzero`
5. `graphchase.scripts.run_pretrain_psro`

To use different settings, edit the `--config` paths and `--set key=value` overrides directly in `run_bash.sh` before running it.


## Citation

If you use GraphChase in your research, please cite our paper:

```bibtex
@inproceedings{zhuang2026graphchase,
  author    = {Shuxin Zhuang and Shuxin Li and Tianji Yang and Muheng Li and
               Xianjie Shi and Bo An and Youzhi Zhang},
  title     = {GraphChase: A Platform and Benchmark for Urban Network Security Games},
  year      = {2026},
  isbn      = {9798400722592},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3770855.3817540},
  doi       = {10.1145/3770855.3817540},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  pages     = {10372--10383},
  numpages  = {12},
  keywords  = {security games, multiplayer games},
  location  = {Republic of Korea},
  series    = {KDD '26}
}
```
