from pathlib import Path

import numpy as np
import cv2

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# my own libs
from unet_model import UNet, UNetModule
from datasets.synthetic_shapes_dataset import SyntheticShapesDataModule



if __name__ == "__main__":

    # sets seeds for numpy, torch and python.random.
    pl.seed_everything(42, workers=True)
    DETERMINISTIC_TRAINER = True
    FAST_DEV_RUN_TRAINER = False

    torch.set_default_tensor_type(torch.FloatTensor)

    BATCHES_PER_EPOCH = 10000
    BATCH_SIZE = 32

    syntheticshapes = SyntheticShapesDataModule(BATCHES_PER_EPOCH, BATCH_SIZE)
    train_loader = syntheticshapes.train_dataloader()
    val_loader = syntheticshapes.val_dataloader()

    # the UNet inspired model
    unet_magicpoint = UNetModule(UNet())

    # train model
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=50, deterministic=DETERMINISTIC_TRAINER,
                         fast_dev_run=FAST_DEV_RUN_TRAINER,
                         default_root_dir="unet_magicpoint_model",
                         callbacks=[DeviceStatsMonitor(), ModelSummary(max_depth=2)])
    trainer.fit(model=unet_magicpoint, train_dataloaders=train_loader)  # val_dataloaders=val_loader

    test_loader = syntheticshapes.test_dataloader()
    trainer.test(model=unet_magicpoint, dataloaders=test_loader)

    if FAST_DEV_RUN_TRAINER == True:
        trainer.save_checkpoint("dummy_model.ckpt")
    else:
        trainer.save_checkpoint("current_best_model.ckpt")


