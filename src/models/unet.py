import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.network_blocks import Conv2DSequence, Conv2DSequenceWithPermanentDropout
from src.utils.module_utils import ModuleIntDict
from src.utils.custom_types import OutputDictType

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )

def double_conv_batch_norm(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=1, batch_norm=False):
        super().__init__()

        base = 32

        if batch_norm:
            conv_block = double_conv_batch_norm
        else:
            conv_block = double_conv

        self.dconv_down1 = conv_block(n_channels, base)
        self.dconv_down2 = conv_block(base, base*2)
        self.dconv_down3 = conv_block(base*2, base*4)
        self.dconv_down4 = conv_block(base*4, base*8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = conv_block(base*4 + base*8, base*4)
        self.dconv_up2 = conv_block(base*2 + base*4, base*2)
        self.dconv_up1 = conv_block(base + base*2, base)

        self.conv_last = nn.Conv2d(base, n_classes, kernel_size=1)

        # Choose activation based on the number of classes
        if n_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([conv3, x], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat([conv2, x], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = torch.cat([conv1, x], dim=1)
        x = self.dconv_up1(x)

        x = self.conv_last(x)

        out = self.activation(x)

        return out



class UNetEncoder(nn.Module):
    def __init__(
        self,
        total_levels: int,
        input_channels: int = 1,
        n0: int = 32,
        permanent_dropout: float = None,
    ):
        super(UNetEncoder, self).__init__()

        if permanent_dropout is not None:

            def conv_sequence(
                in_channels: int,
                out_channels: int,
                depth: int,
                kernel_size: int = 3,
            ):
                return Conv2DSequenceWithPermanentDropout(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    depth=depth,
                    kernel_size=kernel_size,
                    dropout_prob=permanent_dropout,
                )

        else:
            conv_sequence = Conv2DSequence

        self.total_levels = total_levels

        # increase num_channels until the 4th level, then use n0*6 channels
        self.num_channels = {
            k: n0 * v for k, v in enumerate([1, 2, 4] + [6] * (total_levels - 3))
        }

        # Create downsampling layer
        self.downsample = nn.AvgPool2d(
            kernel_size=2, stride=2, padding=0, ceil_mode=True
        )

        # Create layers of main downstream path
        self.down_blocks = ModuleIntDict(
            {
                0: conv_sequence(
                    in_channels=input_channels,
                    out_channels=self.num_channels[0],
                    depth=3,
                )
            }
        )
        for k in range(1, total_levels):
            self.down_blocks[k] = conv_sequence(
                in_channels=self.num_channels[k - 1],
                out_channels=self.num_channels[k],
                depth=3,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> OutputDictType:
        # Going all the way down on the encoding path
        down_activations = {0: self.down_blocks[0](x)}
        for k in range(1, self.total_levels):
            down_sampled = self.downsample(down_activations[k - 1])
            down_activations[k] = self.down_blocks[k](down_sampled)

        return down_activations


class UNetDecoder(nn.Module):
    def __init__(
        self,
        total_levels: int,
        n0: int = 32,
        permanent_dropout: float = None,
    ):
        super(UNetDecoder, self).__init__()

        if permanent_dropout is not None:

            def conv_sequence(
                in_channels: int,
                out_channels: int,
                depth: int,
                kernel_size: int = 3,
            ):
                return Conv2DSequenceWithPermanentDropout(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    depth=depth,
                    kernel_size=kernel_size,
                    dropout_prob=permanent_dropout,
                )

        else:
            conv_sequence = Conv2DSequence

        self.total_levels = total_levels
        self.num_channels = {
            k: n0 * v for k, v in enumerate([1, 2, 4] + [6] * (total_levels - 3))
        }

        # Create layers of main upstream path
        self.up_blocks = ModuleIntDict()
        for k in reversed(range(total_levels - 1)):
            self.up_blocks[k] = conv_sequence(
                in_channels=self.num_channels[k + 1] + self.num_channels[k],
                out_channels=self.num_channels[k],
                depth=3,
            )

    def forward(
        self,
        features: OutputDictType,
    ) -> torch.Tensor:
        intermediate = features[self.total_levels - 1]
        for k in reversed(range(self.total_levels - 1)):
            intermediate = F.interpolate(
                intermediate,
                mode="bilinear",
                scale_factor=2,
                align_corners=True,
            )
            intermediate = torch.cat([features[k], intermediate], axis=1)
            intermediate = self.up_blocks[k](intermediate)

        return intermediate


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int):
        super().__init__()

        self.fcomb = Conv2DSequence(
            in_channels=in_channels,
            out_channels=out_channels,
            depth=3,
            kernel_size=1,
        )

        self.final_prediction_layer = nn.Conv2d(
            in_channels=out_channels,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        fcomb = self.fcomb(x)
        return self.final_prediction_layer(fcomb)


class UNetBase(nn.Module):
    def __init__(
        self,
        total_levels: int,
        input_channels: int = 1,
        n0: int = 32,
        permanent_dropout: float = None,
    ):
        super().__init__()

        self.total_levels = total_levels

        self.encoder = UNetEncoder(
            total_levels=total_levels,
            input_channels=input_channels,
            n0=n0,
            permanent_dropout=permanent_dropout,
        )
        self.decoder = UNetDecoder(
            total_levels=total_levels,
            n0=n0,
            permanent_dropout=permanent_dropout,
        )

    def forward(self, x):
        enc_features = self.encoder(x)
        return self.decoder(enc_features)