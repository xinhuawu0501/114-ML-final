
from lightning.pytorch.cli import LightningCLI
from bert_models import BertClassificationModel
from datamodule import MIMICClassificationDataModule
import torch
import os

torch.set_float32_matmul_precision('medium') # to increase training speed

def main():
    cli = LightningCLI(
        BertClassificationModel,
        MIMICClassificationDataModule,
        save_config_callback=None,   # optional
    )

if __name__ == "__main__":
    main()

    

