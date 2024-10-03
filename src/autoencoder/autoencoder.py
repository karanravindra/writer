import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, img_channels: int, width: int):
        super(Encoder, self).__init__()

        self.conv1 = Depthwise(img_channels, width)

        self.conv2 = Depthwise(width, width * 2)

        self.conv3 = Depthwise(width * 2, width * 4)
        self.conv4 = Depthwise(width * 4, width * 4)
        self.conv5 = Depthwise(width * 4, width * 4)

        self.conv6 = Depthwise(width * 4, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, 2)
        x = self.conv6(x)
        return x


class Decoder(nn.Module):
    def __init__(self, img_channels: int, width: int):
        super(Decoder, self).__init__()

        self.conv1 = Depthwise(4, width * 4)

        self.conv2 = Depthwise(width * 4, width * 4)
        self.conv3 = Depthwise(width * 4, width * 4)
        self.conv4 = Depthwise(width * 4, width * 2)

        self.conv5 = Depthwise(width * 2, width)

        self.conv6 = Depthwise(width, img_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.conv5(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.conv6(x)
        return F.sigmoid(x)


class Depthwise(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3, m=4):
        super(Depthwise, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels
        )
        self.conv2 = nn.Conv2d(in_channels, out_channels * m, 1)
        self.conv3 = nn.Conv2d(out_channels * m, out_channels, 1)

        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)

        x += self.skip(residual)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = nn.Sequential(Encoder(1, 6), Decoder(1, 6))
    summary(
        model,
        (1, 1, 32, 32),
        row_settings=("depth", "var_names"),
        col_names=(
            "output_size",
            "num_params",
            "params_percent",
            "mult_adds",
        ),
    )
