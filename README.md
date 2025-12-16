## Note

1. Since the original data from MIMIC-IV is confidential, please apply for access from physionet. In this repo, we provide pre-processed data instead.
2. Please create seperate virtual environment for phase 1 and phase 2

# Phase 1

## How to initial dev env

- Create venv with `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

## To download pre-processed dataset

- train.csv
  `gdown --id 1mwWzcRDEvr31GURHzq6LbigqC6CkAgvj -O train.csv`

- val.csv
  `gdown --id 121E17J5ddezzmKBwQbGeCM3SYAh1M6i9 -O val.csv`

- test.csv
  `gdown --id 1gFI4qjti5W99-_GBqFSSw8kj6NQLv94z -O test.csv`

- make sure the downloaded files locates in phase1/dataset/

### Training

`python main.py fit --config config.yaml`

### To download checkpoint

`gdown --id 1IRiZzPiPpwlSj4CpRZPugzEevCU6utKW -O model_best.ckpt`

- make sure checkpoint file locates in phase1/

### Testing

`python main.py test --ckpt_path "model_best.ckpt" --config config.yaml`

# Phase 2

## How to initial dev env

- Create venv with `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

## Data Preprocessing

The preprocessed data is `phase2/stage2_input_val.csv`.

## Download checkpoint

`gdown --id 19KjEPrl-vLXnpVPA1W8t1gmBwZ_R3w7R -O stage2_train_v6.pth`

- make sure check point locates in phase2/

## Execution

Run the training script with command `python3 stage2_train_v6.py train` .
Run the evaluation script with command `python3 stage2_train_v6.py eval` .

### Configuration

Parameters can be modified directly within the `stage2_train_v6.py` file.

### Outputs

Upon execution, the script will generate:

1.  **Visualizations:** Two `.png` files (`loss_breakdown_v6.png` and `figure_threshold.png`).
2.  **Metrics:** A performance table displayed in the terminal.

### Others

combined_theta_trend_v6.png represents the output of the Stage 2 model trained on input from a Stage 1 model without class re-weighting or gradient clipping.
