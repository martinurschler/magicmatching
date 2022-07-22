import torch
import pytorch_lightning as pl
from torchvision import transforms


class DoubleConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(torch.nn.Module):

    def __init__(self, in_channels=1, out_channels=1, features = [32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.downs = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(torch.nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = torch.nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        channel_dimension = len(x.shape) - 3 # if we have batches (4 dim tensor) this is 1, if not (3dim) 0

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=channel_dimension)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class MagicPointUNetModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = UNet(in_channels=1, out_channels=1)
        self.save_hyperparameters()

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        img, target = batch
        target_hat = self.unet(img)
        train_loss = torch.nn.functional.mse_loss(target_hat, target)
        return train_loss

    #def validation_step(self, batch, batch_idx):
    #    img, target = batch
    #    target_hat = self.unet(img)
    #    val_loss = torch.nn.functional.mse_loss(target_hat, target)
    #    self.log("val loss", val_loss)

    def test_step(self, batch, batch_idx):
        img, target = batch
        target_hat = self.unet(img)
        test_loss = torch.nn.functional.mse_loss(target_hat, target)
        self.log("test loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer