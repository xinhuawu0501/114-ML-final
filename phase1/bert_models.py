import os
import lightning.pytorch as pl
import torch
from torchmetrics.functional.classification import auroc as auroc_f
from torchmetrics.functional.classification import average_precision, accuracy
from torchmetrics.classification import MulticlassAveragePrecision
import transformers
from transformers import AutoModel
import torch.nn.functional as F
import pandas as pd

from loss import FocalLoss

class BertClassificationModel(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 encoder_model_name,
                 warmup_steps,
                 decay_steps,
                 num_training_steps,
                 weight_decay,
                 optimizer_name,
                 criterion_name,
                 freeze_bert,
                 label_smoothing
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        self.encoder.pooler = None
        self.num_classes = num_classes
        self.classification_layer = torch.nn.Linear(768, num_classes)
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.num_training_steps = num_training_steps
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.auroc = None
        self.mean_precision = None
        self.val_output_list = []
        self.test_output_list = []
        self.running_allocated_memory = 0
        self.running_reserved_memory = 0
        self.best_score = 0
        self.label_smoothing = label_smoothing
        self.a_prec_list = []
        self.a_prec_metric = MulticlassAveragePrecision(num_classes=self.num_classes, average='none') # for plotting score for each class
        self.criterion_name = criterion_name
        self.is_focal = criterion_name == 'focal'
        self.lr = 1e-4 if self.is_focal else 2e-5

        self.criterion = FocalLoss(gamma=2.0, reduction='mean') if self.is_focal else None

        if not freeze_bert:
            self.encoder.train()

    def forward(self,
                input_ids,
                attention_mask):
        encoded = self.encoder(input_ids, attention_mask, return_dict=True)[
            'last_hidden_state'][:, 0]
        logits = self.classification_layer(encoded)
        return logits
    
    def on_fit_start(self) -> None:
        print(self.hparams)
        self.a_prec_list.clear()
        self.val_output_list.clear()
        self.a_prec_metric.reset()

        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.watch(self, log="all", log_freq=100)        
        return super().on_fit_start()
    
    # Optional: to log total gradient 
    # def on_before_optimizer_step(self, optimizer):
    #     # 1. Compute the L2 norm of all gradients
    #     #    (We use max_norm=inf so it doesn't actually clip here, just measures)
    #     grad_norm = torch.linalg.vector_norm(
    #         torch.stack([torch.linalg.vector_norm(p.grad.detach(), 2) 
    #                      for p in self.parameters() if p.grad is not None]), 
    #         2
    #     )
        
    #     # 2. Log it to W&B
    #     self.log("Gradients/total_norm", grad_norm, on_step=True, prog_bar=True, logger=True)
        
    #     # 3. Call the parent method (required)
    #     super().on_before_optimizer_step(optimizer)

    def training_step(self, batch, batch_idx):
        # print(f'training use class weight: {self.trainer.datamodule.classes_weight}')
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]
        logits = self.classification_layer(encoded)
        device = logits.device

        _, labels = torch.max(batch['labels'], dim=1)
        labels = labels.to(device)

        if self.is_focal:
            loss = self.criterion(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels, 
                                    weight=self.trainer.datamodule.classes_weight.to(device),
                                    label_smoothing=(0.1 if self.label_smoothing else 0.0),
                                   )
            
        self.log("Train/Loss", loss)
        return loss

    def test_step(self, batch, batch_idx, **kwargs):
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]
        logits = self.classification_layer(encoded)
        probs = torch.softmax(logits, dim=1)

        output = {"hadm_ids": batch["hadm_ids"], "logits": logits, "labels": batch["labels"], "probs": probs}
        self.test_output_list.append(output)

        return output
    
    def on_test_epoch_end(self) -> None:
        device = self.device
        logits = torch.cat([x["logits"] for x in self.test_output_list])
        labels = torch.cat([x["labels"] for x in self.test_output_list]).int().to(device)
        probs = torch.cat([x["probs"] for x in self.test_output_list])
        ids = torch.cat([x["hadm_ids"] for x in self.test_output_list]).cpu()


        _, labels = torch.max(labels, dim=1)
        auroc_ = auroc_f(logits, labels, num_classes=self.num_classes, 
                                    task="multiclass", average='macro')
        mean_precision_ = average_precision(logits, labels, num_classes=self.num_classes, 
                                    task="multiclass", average='macro')
        accuracy_ = accuracy(logits, labels, task='multiclass', num_classes=self.num_classes,
                                average='macro') 
        if self.is_focal:
            loss = self.criterion(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels)

        if mean_precision_ > self.best_score: # type: ignore
            self._store_pred(ids=ids, labels=labels, probs=probs, dataset_name="test")
            self.best_score = mean_precision_
            print(f'save best model with {mean_precision_}')

        # store a_prec score for each class
        self.a_prec_list.append(self.a_prec_metric(logits, labels).detach().cpu())
     
        self.log("Test/loss", loss)
        self.log("Test/AUROC", auroc_)
        self.log("Test/A_PREC", mean_precision_)

        self.test_output_list = list()

    def validation_step(self, batch, batch_idx, **kwargs):
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)[
            'last_hidden_state'][:, 0]
        logits = self.classification_layer(encoded)
        probs = torch.softmax(logits, dim=1)
        output = {"hadm_ids": batch["hadm_ids"], "logits": logits, "labels": batch["labels"], "probs": probs}

        self.val_output_list.append(output)

        return output
    
    def on_validation_epoch_end(self) -> None:
        device = self.device
        logits = torch.cat([x["logits"] for x in self.val_output_list])
        labels = torch.cat([x["labels"] for x in self.val_output_list]).int().to(device)
        probs = torch.cat([x["probs"] for x in self.val_output_list])
        ids = torch.cat([x["hadm_ids"] for x in self.val_output_list]).cpu()

        _, labels = torch.max(labels, dim=1)
        auroc_ = auroc_f(logits, labels, num_classes=self.num_classes, 
                                    task="multiclass", average='macro')
        mean_precision = average_precision(logits, labels, num_classes=self.num_classes, 
                                    task="multiclass", average='macro')
        accuracy_ = accuracy(logits, labels, task='multiclass', num_classes=self.num_classes,
                                average='macro') 
        
        if self.is_focal:
            loss = self.criterion(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels)

        if self.trainer.sanity_checking:
            print("Sanity check passed. Discarding metrics.")
            self.val_output_list.clear()
            return 

        
        if mean_precision > self.best_score:
            self._store_pred(ids=ids, labels=labels, probs=probs, dataset_name="val")
            self.best_score = mean_precision
            print(f'save best model with {mean_precision}')
        
        self.a_prec_list.append(self.a_prec_metric(logits, labels).detach().cpu())
        self.plot_curve()
        
        self.log("Val/loss", loss)
        self.log("Val/AUROC", auroc_)
        self.log("Val/A_PREC", mean_precision)
        self.log("Val/BalancedAcc", accuracy_)
        self.val_output_list.clear()
        
    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        weight_decay = 0.01
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                weight_decay
        }, {
            'params':
                [p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, self.warmup_steps, num_training_steps=self.num_training_steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
        }

        return [optimizer], [scheduler]
    
    def _store_pred(self, ids, labels, probs, dataset_name = "train"):
        # Convert to CPU numpy
        probs_np = probs.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Build dataframe
        df = pd.DataFrame(probs_np, columns=[f"class_{i}_prob" for i in range(self.num_classes)])
        df["true_label"] = labels_np
        df["hadm_id"] = ids
        df.set_index("hadm_id")

        # Save CSV
        output_dir = f"../model_output/{dataset_name}"
        self.filename = ""
        try:
     
            if self.trainer.logger.version:
                self.filename = self.trainer.logger.version
    
  
        except Exception as e:
            print(e)
        
        full_path = os.path.join(output_dir, self.filename) if self.filename else output_dir
        os.makedirs(full_path, exist_ok=True)


        output_path = os.path.join(full_path, "predictions.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {dataset_name} predictions to: {output_path}")

    def plot_curve(self):
        if not len(self.a_prec_list): return
        fig, ax = self.a_prec_metric.plot(self.a_prec_list)

        ax.set_title("A_PREC per class")
        ax.set_xlabel("Epoch")
        fig.savefig(f"{self.filename}_ap_plot.png")
        return
    
    def teardown(self, stage: str) -> None:
        try:
            self.plot_curve()
        except Exception as e:
            print(e)
        return super().teardown(stage)