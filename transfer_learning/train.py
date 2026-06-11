import os
import pytorch_lightning as pl 
from tqdm import tqdm
import warnings
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from .restnet18 import ResNetFinetuner
from .dataset import CatDogDataset
from .utils import set_seed, train_transforms, val_transforms
warnings.filterwarnings("ignore")
os.environ["WANDB_API_KEY"] = "YOUR API KEY"

def train(train_dataloader, val_dataloader,  id2label, label2id, seed=42): 
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training but no GPU was found."
        )
    set_seed(seed=seed)
    resnet18_finetuner = ResNetFinetuner(id2label=id2label, label2id=label2id, 
                                        train_dataloader=train_dataloader, val_dataloader=val_dataloader, 
                                        lr=1e-4, metrics_interval=50)

    wandb_logger = WandbLogger(project='Cats-Dogs-Classification', name='resnet18')

    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=5, 
        verbose=True, 
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(filename='best-{epoch:02d}-{val_loss:.4f}', 
                                        save_top_k=1, monitor="val_loss", mode="min", verbose=True)

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator='gpu',
        devices=1,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=30,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.1,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    trainer.fit(resnet18_finetuner)

if __name__ == '__main__': 
    BATCH_SIZE = 64
    label2id = {'cat': 0, 'dog': 1}
    id2label = {v: k for k, v in label2id.items()}

    train_dataset = CatDogDataset(rootDir='dogs-cats-10k/train', 
                                  label2id=label2id,
                                  transforms=train_transforms)
    val_dataset = CatDogDataset(rootDir='dogs-cats-10k/val', 
                                label2id=label2id,
                                transforms=val_transforms)
    test_dataset = CatDogDataset(rootDir='/dogs-cats-10k/test',
                                label2id=label2id,
                                transforms=val_transforms)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Start training
    train(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
          id2label=id2label, label2id=label2id)