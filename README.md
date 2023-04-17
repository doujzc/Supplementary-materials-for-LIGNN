# Link Level Implicit Graph Neural Networks

This is the official implementation of Link Level Implicit Graph Neural Networks (LIGCN).

## Requirements
- pytorch
- torch-geometric
- tqdm
- numpy
- sklearn

## Synthetic experiments
To run the synthetic experiments
```console
python train_synthetic.py
```

## PCQM-Contact
The dataset is too big to be provided in this folder. Please first follow the instructions in Long Range Graph Benchmark and download the PCQM-Contact files. Then, put the `raw` folder (containing `train.pt`, `valid.pt`, `test.pt`) into `./data/pcqm-contact/raw`.

To run the molecular property prediction experiments
```console
python train_pcqm.py
```

## Knowledge graph completion
To run the KGC experiments
```console
python train_kg.py --dataset <dataset name>
```
For example,
```console
python train_kg.py --dataset GraIL-BM_WN18RR_v1 --out_bias --cuda
python train_kg.py --dataset GraIL-BM_fb237_v1 --cuda
python train_kg.py --dataset GraIL-BM_nell_v1 --cuda
```