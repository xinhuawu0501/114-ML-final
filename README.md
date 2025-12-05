# Phase 1

## How to initial dev env

- Navigate into `phase1/`
- Create venv with `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

# File structure:

- ml-final_final_ver/
  - phase1/
    - utils/
      - data_utils.py
      - mimic_utils.py
    - main.py
    - bert_models.py
    - datamodule.py
    - custom_metrics.py
  - dataset/
    - hosp/
      - transfers.csv.gz
      - admissions.csv.gz
    - note/
      - discharge.csv
    - pr/
  - requirements.txt

## To create csv file for training:

- make sure `dataset/` directory contains necessary files and is structured as above
- navigate into `src/utils/`
- run `create_pr_data_csv` function in `data_utils.py`. This should generate 3 files: train.csv/val.csv/test.csv under `pr/` directory

### Training

`python main.py fit --config config.yaml`

### Testing

`python main.py test --ckpt_path "../checkpoint/model_best.ckpt" --config config.yaml`

# Phase 2
