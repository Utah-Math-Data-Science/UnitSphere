# SCHull Respository

This is the official repository for the SCHull models. 


## Installation

To install the requirements for the SCHull models, you can use the following command:

```bash
pip install -r requirements.txt
```
## Usage

All datasets are downloaded and processed at runtime. They are stored in a `/root/workspace/data/` directory.

**Synthetic Experiments**

The synthetic experiments $n$-chains and NestedSquares can be found in the `/docs/` directory. These jupyter notebooks contain the code to generate the results in the paper.


**Benchmark Experiments**

The benchmark experiments on molecule datasets can be run using either of the two following commands:

```bash
python main_qm9.py
python main_md17.py
```

The benchmark experiments on molecule datasets can be run using either of the two following commands:

```bash
python main_fold_reat.py --dataset [DATASET]
python main_lba.py
```
