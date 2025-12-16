## How to initial dev env

- Create venv with `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

# Phase 1

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

- make sure checkpoint file locates in checkpoint/

### Testing

`python main.py test --ckpt_path "../checkpoint/model_best.ckpt" --config config.yaml`

# Phase 2

## Data Preprocessing

Run `python3 preprocess_data.py` to prepare the data.
This script transforms the raw prediction file from `model_output/val/predictions.csv` into the input format required for Stage 2 at `phase2/stage2_input_val.csv`.

## Download checkpoint

`gdown --id 19KjEPrl-vLXnpVPA1W8t1gmBwZ_R3w7R -O stage2_train_v6.pth`

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
