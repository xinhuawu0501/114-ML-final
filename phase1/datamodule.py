import json
import random
from typing import Optional
import csv
import lightning.pytorch as pl
from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ClassifcationCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_seq_len: int = 512,
        all_examples_with_null: bool = False,

    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.all_examples_with_null = all_examples_with_null
        

    def __call__(self, data):
        admission_notes = [x["text"] for x in data]
        labels = torch.stack([x["labels"] for x in data])
        hadm_ids = torch.tensor([x["hadm_id"] for x in data])

        tokenized = self.tokenizer(
            admission_notes,
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
        )

        input_ids = [torch.tensor(x) for x in tokenized["input_ids"]]
        attention_masks = [
            torch.tensor(x, dtype=torch.bool) for x in tokenized["attention_mask"]
        ]
        lengths = torch.tensor([len(x) for x in input_ids])
      
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
     

        tokens = [x.tokens for x in tokenized.encodings]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "lengths": lengths,
            "tokens": tokens,
            "hadm_ids": hadm_ids
        }


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, examples, label_lookup, sampling_strategy: str = "weighted"):
        # tokenize admission notes
        self.examples = examples
        self.label_lookup = label_lookup
        self.inverse_label_lookup = {v: k for k, v in label_lookup.items()}
        self.sampling_strategy = sampling_strategy
        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples.loc[idx]
        note = example["text"]
        labels = example["labels"]

        label_arr = torch.zeros(len(self.label_lookup), dtype=torch.float32)
        label_arr[self.label_lookup[labels]] = 1
        return {
            "text": note,
            "labels": label_arr,
            "hadm_id": example["hadm_id"]
        }


class MIMICClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        use_code_descriptions: bool = False,
        data_dir: str = "../dataset/",
        task: str = "pr",
        batch_size: int = 32,
        eval_batch_size: int = 16,
        tokenizer_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        num_workers: int = 8,
        sampling_strategy: str = "random",
        val_sampling_strategy: str = "random",
        max_seq_len: int = 512,
    ):
        super().__init__()

        test_data = pd.read_csv(data_dir + task + "/test.csv").rename(
            columns={"class": "labels"}
        )
        train_data = pd.read_csv(data_dir + task + "/train.csv").rename(
            columns={"class": "labels"}
        )
        val_data = pd.read_csv(data_dir + task + "/val.csv").rename(
            columns={"class": "labels"}
        )
        self.training_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        all_labels = list(train_data.labels.unique())
        all_labels.sort()
        
        label_idx = {v: k for k, v in enumerate(all_labels)}
        training_data_labels = [x for x in train_data.labels]

        label_counts = {}
        for l in training_data_labels:
            k = str(l)
            if k not in label_counts: label_counts[k] = 0
            label_counts[k] += 1
        
        
        # sklearn util
        # classes_weight = compute_class_weight(
        #     class_weight="balanced",
        #     classes=np.array(all_labels), # unique classes 
        #     y=training_data_labels # actual classes for all data
        # )


        # effective number of sample
        ttl_data = len(training_data_labels)
        counts = torch.tensor([label_counts[str(i)] for i in range(1, len(all_labels) + 1)], dtype=torch.float)
        print(f'Total training data: {ttl_data}\nClass counts: {counts}')

        beta = 0.9999
        effective_num = (1.0 - torch.pow(beta, counts))/ (1.0 - beta)

        weights = 1.0 / effective_num
        weights = weights / weights.sum() * len(counts)

        self.classes_weight = weights.float()
        rounded = torch.round(self.classes_weight, decimals=2)
        print(f"store classes_weight: {rounded}")

        # build label index
        self.label_idx = label_idx
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.collator = ClassifcationCollator(self.tokenizer, max_seq_len)
        self.num_workers = num_workers
        self.task = task
        self.sampling_strategy = sampling_strategy
        self.val_sampling_strategy = val_sampling_strategy


    def setup(self, stage: Optional[str] = None):
        mimic_train = ClassificationDataset(
            self.training_data,
            label_lookup=self.label_idx,
            sampling_strategy=self.sampling_strategy,
        )
        
        mimic_val = ClassificationDataset(
            self.val_data,
            label_lookup=self.label_idx,
            sampling_strategy=self.val_sampling_strategy,
        )

        mimic_test = ClassificationDataset(
            self.test_data,
            label_lookup=self.label_idx,
            sampling_strategy=self.val_sampling_strategy,
        )
        self.mimic_train = mimic_train
        self.mimic_val = mimic_val
        self.mimic_test = mimic_test
        print("Val length: ", len(self.mimic_val))
        print("Train Length: ", len(self.mimic_train))

    def train_dataloader(self):
        return DataLoader(
            self.mimic_train,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mimic_val,
            batch_size=self.eval_batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mimic_test,
            batch_size=self.eval_batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
