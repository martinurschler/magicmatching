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
            #print("\ndown before conv", x.shape)
            x = down(x)
            #print("down after conv", x.shape)
            skip_connections.append(x)
            x = self.pool(x)
            #print("down after pool", x.shape)

        x = self.bottleneck(x)
        #print("bottleneck", x.shape)
        skip_connections = skip_connections[::-1]

        #print(x.shape, len(x.shape))
        channel_dimension = len(x.shape) - 3 # if we have batches (4 dim tensor) this is 1, if not (3dim) 0

        for idx in range(0, len(self.ups), 2):
            #print("up before transp conv", x.shape)
            x = self.ups[idx](x)
            #print("up after transpose conv", x.shape)
            skip_connection = skip_connections[idx//2]
            #print("skip shape", skip_connection.shape)
            if x.shape != skip_connection.shape:
                #print("resize")
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])
                #print("after resize", x.shape)
            concat_skip = torch.cat((skip_connection, x), dim=channel_dimension)
            #print("after cat", concat_skip.shape)
            x = self.ups[idx+1](concat_skip)
            #print("x shape", x.shape)

        return self.final_conv(x)

class UNetModule(pl.LightningModule):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.save_hyperparameters()

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        img, target = batch
        target_hat = self.unet(img)
        train_loss = torch.nn.functional.mse_loss(target_hat, target)
        return train_loss

    def validation_step(self, batch, batch_idx):
        img, target = batch
        target_hat = self.unet(img)
        val_loss = torch.nn.functional.mse_loss(target_hat, target)
        self.log("val loss", val_loss)

    def test_step(self, batch, batch_idx):
        img, target = batch
        target_hat = self.unet(img)
        test_loss = torch.nn.functional.mse_loss(target_hat, target)
        self.log("test loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer