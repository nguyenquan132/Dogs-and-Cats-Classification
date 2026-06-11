import torch
import pytorch_lightning as pl 
from transformers import ResNetForImageClassification
from torchmetrics import Accuracy, F1Score, Precision, Recall

class ResNetFinetuner(pl.LightningModule):
    def __init__(self, id2label, label2id, train_dataloader=None, val_dataloader=None, 
                 lr=None, metrics_interval=100):
        super(ResNetFinetuner, self).__init__()
        self.save_hyperparameters(ignore=['train_dataloader', 'val_dataloader'])
        self.id2label = id2label
        self.label2id = label2id
        self.num_classes = len(id2label.keys())
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.lr = lr
        self.metrics_interval = metrics_interval
        self.model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18",
                                                                  num_labels=len(id2label.keys()),
                                                                  id2label=self.id2label, 
                                                                  label2id=self.label2id,
                                                                  ignore_mismatched_sizes=True)
        self._freeze_backbone()
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_f1 = F1Score(task='multiclass', num_classes=self.num_classes, average='macro')
        self.val_precision = Precision(task='multiclass', num_classes=self.num_classes, average='macro')
        self.val_recall = Recall(task='multiclass', num_classes=self.num_classes, average='macro')

    def _freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values=pixel_values,
                            labels=labels)
        return outputs

    def training_step(self, batch, batch_idx): 
        pixel_values = batch[0]
        labels = batch[1]
        batch_size = len(labels)
        outputs = self(pixel_values=pixel_values, labels=labels)
        loss, preds = outputs.loss, outputs.logits.argmax(dim=1)
        if batch_idx % self.metrics_interval == 0:
            self.log('train_loss', loss, batch_size=batch_size, prog_bar=True, logger=True)
        self.train_acc(preds, labels)
        return loss

    def on_training_epoch_end(self):
        train_acc_value = self.train_acc.compute()
        self.log('train_acc', train_acc_value, prog_bar=True, logger=True)  
        self.train_acc.reset()
        
    def validation_step(self, batch, batch_idx): 
        pixel_values = batch[0]
        labels = batch[1]
        batch_size = len(labels)
        outputs = self(pixel_values=pixel_values, labels=labels)
        loss, preds = outputs.loss, outputs.logits.argmax(dim=1)
        self.log('val_loss', loss, batch_size=batch_size, prog_bar=True, logger=True)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        return loss
        
    def on_validation_epoch_end(self):
        val_acc_value = self.val_acc.compute()
        val_f1_value = self.val_f1.compute()
        val_precision_value = self.val_precision.compute()
        val_recall_value = self.val_recall.compute()
        
        self.log('val_acc', val_acc_value, prog_bar=True, logger=True)
        self.log('val_f1', val_f1_value, prog_bar=True, logger=True)
        self.log('val_precision', val_precision_value, prog_bar=False, logger=True)
        self.log('val_recall', val_recall_value, prog_bar=False, logger=True)
        
        # Reset
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    def train_dataloader(self): 
        return self.train_dl
    def val_dataloader(self): 
        return self.val_dl