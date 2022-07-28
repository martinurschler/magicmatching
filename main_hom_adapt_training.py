from pathlib import PureWindowsPath, Path

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# my own libs
from unet_model import MagicPointUNetModule
from datasets.coco_dataset import CocoWithHomAdaptDataModule



if __name__ == "__main__":

    # sets seeds for numpy, torch and python.random.
    pl.seed_everything(42, workers=True)
    DETERMINISTIC_TRAINER = True
    FAST_DEV_RUN_TRAINER = False

    torch.set_default_tensor_type(torch.FloatTensor)

    BATCHES_PER_EPOCH = 10000
    BATCH_SIZE = 32

    coco_base_path = PureWindowsPath("E:/01_Repos/xzho372/data/COCO")

    print("loading magicpoint model as starting point for training")
    model_path = str(Path("data") / "pretrained_archive" / "current_best_magicpoint_model_v1.ckpt")
    unet_superpoint = MagicPointUNetModule.load_from_checkpoint(model_path)

    coco_hom_adapt = CocoWithHomAdaptDataModule(data_dir=coco_base_path, absolute_path_of_magicpoint=model_path)
    train_loader = coco_hom_adapt.train_dataloader()

    # train model
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=50, deterministic=DETERMINISTIC_TRAINER,
                         fast_dev_run=FAST_DEV_RUN_TRAINER,
                         default_root_dir="unet_superpoint_model_v0",
                         callbacks=[DeviceStatsMonitor(), ModelSummary(max_depth=2)])
    trainer.fit(model=unet_superpoint, train_dataloaders=train_loader)

    if FAST_DEV_RUN_TRAINER == True:
        trainer.save_checkpoint("dummy_model.ckpt")
    else:
        trainer.save_checkpoint("current_best_superpoint_model_v1.ckpt")


