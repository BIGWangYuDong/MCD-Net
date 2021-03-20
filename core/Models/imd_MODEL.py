import torch
import torch.nn as nn


class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=4):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels // distillation_rate)             # 32/3 = 8
        self.remaining_channels_3 = int(in_channels - self.distilled_channels)      # 32-8 = 24
        self.remaining_channels_2 = int(self.remaining_channels_3 - self.distilled_channels)    # 24-8=16
        self.remaining_channels_1 = int(self.distilled_channel2 - self.distilled_channels)      # 16-8=8

        self.conv0 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv1 = nn.Conv2d(self.remaining_channels_3, self.remaining_channels_3, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.remaining_channels_2, self.remaining_channels_2, 3, 1, 1)
        self.conv3 = nn.Conv2d(self.remaining_channels_1, self.remaining_channels_1, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0(x))

        distilled_c1, remaining_c1 = torch.split(out, (self.distilled_channels, self.remaining_channels_3), dim=1)

        out_c2 = self.relu(self.conv1(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels_2), dim=1)

        out_c3 = self.relu(self.conv2(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels_1), dim=1)

        out_c4 = self.relu(self.conv3(remaining_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out = self.relu(self.conv4(out))

        return out