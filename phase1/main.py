
from lightning.pytorch.cli import LightningCLI
from bert_models import BertClassificationModel
from datamodule import MIMICClassificationDataModule
import torch
import pandas as pd
import os
from utils.data_utils import misclassified_ratio

torch.set_float32_matmul_precision('medium') # to increase training speed

def main():
    cli = LightningCLI(
        BertClassificationModel,
        MIMICClassificationDataModule,
        save_config_callback=None,   # optional
    )

if __name__ == "__main__":
    main()

    # load prediction
    # mode = ["val", "test", "train"]
    # model_path = os.path.join("./model_output", mode[0], "predictions.csv")
    # pred = pd.read_csv(model_path, index_col="hadm_id")

    # misclassified_ratio(prediction=pred)

    

