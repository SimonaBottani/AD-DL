# coding: utf8

from .modules import PadMaxPool3d, Flatten
import torch.nn as nn
import torch
import torch.nn.functional as F


"""
All the architectures are built here
"""
class AddingNodes(nn.Module):
    def forward(self, input, nodes):
        return torch.cat([input, nodes], dim=1)

class Conv5_FC3(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv5_FC3_MultiTask(nn.Module):
    """
    Classifier for a multi-task classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_MultiTask, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier1 = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, 4)
        )

        self.classifier2 = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, 3)
        )

        self.classifier3 = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )



        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x) # the same for all the tasks
        out1 = self.classifier1(x)
        out2 = self.classifier3(x)
        out3 = self.classifier2(x)
        out4 = self.classifier2(x)

        return out1, out2, out3, out4

class Conv5_FC3_MultiTask_2_2(nn.Module):
    """
    Classifier for a multi-task classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_MultiTask_2_2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier3 = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )



        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x) # the same for all the tasks
        out1 = self.classifier3(x)
        out2 = self.classifier3(x)

        return out1, out2

class Conv5_FC3_MultiTask_3_2(nn.Module):
    """
    Classifier for a multi-task classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_MultiTask_3_2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier3 = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )



        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x) # the same for all the tasks
        out1 = self.classifier3(x)
        out2 = self.classifier3(x)
        out3 = self.classifier3(x)

        return out1, out2, out3

class Conv5_FC3_MultiTask_4_2(nn.Module):
    """
    Classifier for a multi-task classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_MultiTask_4_2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier3 = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x) # the same for all the tasks
        out1 = self.classifier3(x)
        out2 = self.classifier3(x)
        out3 = self.classifier3(x)
        out4 = self.classifier3(x)

        return out1, out2, out3, out4

class Conv5_FC3_4_classes(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_4_classes, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 4)

        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

class Conv5_FC3_3_classes(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_3_classes, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 3)

        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        (x==x)


        return x


class Conv5_FC3_mni(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Extensive preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_mni, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 4 * 5 * 4, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 128, 4, 5, 4]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

class UNetDown(nn.Module):
    """Descending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    def __init__(self, in_size, out_size):
        super(UNetDown, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_size, out_size, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2)
          )

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Ascending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    import torch
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose3d(in_size, out_size, kernel_size=4,
                               stride=2, padding=1),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x


class FinalLayer(nn.Module):
    """Final block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    import torch
    def __init__(self, in_size, out_size):
        super(FinalLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_size, out_size, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout=0.5):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)

        self.final = FinalLayer(128, 1)

    def forward(self, x):
        print(x.shape)
        print('zero padding')
        source = torch.zeros(256, 256, 256)
        x[:169, :208, :179] = source
        print(x.shape)
        d1 = self.down1(x)
        print(d1.shape)
        d2 = self.down2(d1)
        print(d2.shape)
        d3 = self.down3(d2)
        print(d3.shape)
        d4 = self.down4(d3)
        print(d4.shape)
        d5 = self.down5(d4)
        print(d5.shape)
        print('U1')

        u1 = self.up1(d5)
        print(u1.shape)
        print(d4.shape)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)

        return self.final(u4, d1)



class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv3d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm3d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv3d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm3d(n3x3red),
            nn.ReLU(True),
            nn.Conv3d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv3d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm3d(n5x5red),
            nn.ReLU(True),
            nn.Conv3d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5),
            nn.ReLU(True),
            nn.Conv3d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool3d(3, stride=1, padding=1),
            nn.Conv3d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm3d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes=2):
        super(InceptionAux, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 128, kernel_size=1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
        )
        self.fc1 = nn.Linear(3456, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool3d(x, (3, 3, 3))
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc2(x)

        return x


class GoogLeNet3D_new(nn.Module):
    def __init__(self, training=True, **kwargs):
        super(GoogLeNet3D_new, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            nn.MaxPool3d(3, stride=2),

            nn.Conv3d(64, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            nn.Conv3d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(True),

            nn.MaxPool3d(3, stride=2),
        )

        self.training = training

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool3d(3, stride=2)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool3d((2, 3, 2), stride=1)
        self.linear = nn.Linear(27648, 2)

        self.aux1 = InceptionAux(512, 2)
        self.aux2 = InceptionAux(528, 2)

    def forward(self, x, covars = None):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)

        # For gradient injection
        if self.training:
            aux_out1 = self.aux1(out)

        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)

        # For gradient injection
        if self.training:
            aux_out2 = self.aux2(out)

        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = F.dropout(out, 0.4, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.training:
            return out, aux_out1, aux_out2
        return out

class ResBlock(nn.Module):
    def __init__(self, block_number, input_size):
        super(ResBlock, self).__init__()

        layer_in = input_size if input_size is not None else 2 ** (block_number + 1)
        layer_out = 2 ** (block_number + 2)

        self.conv1 = nn.Conv3d(layer_in, layer_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(layer_out)
        self.act1 = nn.ELU()

        self.conv2 = nn.Conv3d(layer_out, layer_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(layer_out)

        # shortcut
        self.shortcut = nn.Sequential(
            nn.Conv3d(layer_in, layer_out, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.act2 = nn.ELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.act2(out)
        return out


from .modules import PadMaxPool3d, Flatten
import torch.nn as nn
import torch
import torch.nn.functional as F


"""
All the architectures are built here
"""
class AddingNodes(nn.Module):
    def forward(self, input, nodes):
        return torch.cat([input, nodes], dim=1)

class Conv5_FC3(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv5_FC3_MultiTask(nn.Module):
    """
    Classifier for a multi-task classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_MultiTask, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier1 = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, 4)
        )

        self.classifier2 = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, 3)
        )

        self.classifier3 = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )



        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x) # the same for all the tasks
        out1 = self.classifier1(x)
        out2 = self.classifier3(x)
        out3 = self.classifier2(x)
        out4 = self.classifier2(x)

        return out1, out2, out3, out4

class Conv5_FC3_MultiTask_2_2(nn.Module):
    """
    Classifier for a multi-task classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_MultiTask_2_2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier3 = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )



        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x) # the same for all the tasks
        out1 = self.classifier3(x)
        out2 = self.classifier3(x)

        return out1, out2

class Conv5_FC3_MultiTask_3_2(nn.Module):
    """
    Classifier for a multi-task classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_MultiTask_3_2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier3 = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )



        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x) # the same for all the tasks
        out1 = self.classifier3(x)
        out2 = self.classifier3(x)
        out3 = self.classifier3(x)

        return out1, out2, out3

class Conv5_FC3_MultiTask_4_2(nn.Module):
    """
    Classifier for a multi-task classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_MultiTask_4_2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier3 = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x) # the same for all the tasks
        out1 = self.classifier3(x)
        out2 = self.classifier3(x)
        out3 = self.classifier3(x)
        out4 = self.classifier3(x)

        return out1, out2, out3, out4

class Conv5_FC3_4_classes(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_4_classes, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 4)

        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

class Conv5_FC3_3_classes(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Minimal preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_3_classes, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 3)

        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        (x==x)


        return x


class Conv5_FC3_mni(nn.Module):
    """
    Classifier for a binary classification task

    Image level architecture used on Extensive preprocessing
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_mni, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 4 * 5 * 4, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 128, 4, 5, 4]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

class UNetDown(nn.Module):
    """Descending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    def __init__(self, in_size, out_size):
        super(UNetDown, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_size, out_size, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(out_size),
            nn.LeakyReLU(0.2)
          )

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Ascending block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    import torch
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose3d(in_size, out_size, kernel_size=4,
                               stride=2, padding=1),
            nn.InstanceNorm3d(out_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x


class FinalLayer(nn.Module):
    """Final block of the U-Net.

    Args:
        in_size: (int) number of channels in the input image.
        out_size : (int) number of channels in the output image.

    """
    import torch
    def __init__(self, in_size, out_size):
        super(FinalLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_size, out_size, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dropout=0.5):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 64)

        self.final = FinalLayer(128, 1)

    def forward(self, x):
        print(x.shape)
        print('zero padding')
        source = torch.zeros(256, 256, 256)
        x[:169, :208, :179] = source
        print(x.shape)
        d1 = self.down1(x)
        print(d1.shape)
        d2 = self.down2(d1)
        print(d2.shape)
        d3 = self.down3(d2)
        print(d3.shape)
        d4 = self.down4(d3)
        print(d4.shape)
        d5 = self.down5(d4)
        print(d5.shape)
        print('U1')

        u1 = self.up1(d5)
        print(u1.shape)
        print(d4.shape)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)

        return self.final(u4, d1)



class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv3d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm3d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv3d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm3d(n3x3red),
            nn.ReLU(True),
            nn.Conv3d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv3d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm3d(n5x5red),
            nn.ReLU(True),
            nn.Conv3d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5),
            nn.ReLU(True),
            nn.Conv3d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool3d(3, stride=1, padding=1),
            nn.Conv3d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm3d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes=2):
        super(InceptionAux, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 128, kernel_size=1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
        )
        self.fc1 = nn.Linear(3456, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool3d(x, (3, 3, 3))
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc2(x)

        return x


class GoogLeNet3D_new(nn.Module):
    def __init__(self, training=True, **kwargs):
        super(GoogLeNet3D_new, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            nn.MaxPool3d(3, stride=2),

            nn.Conv3d(64, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            nn.Conv3d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(True),

            nn.MaxPool3d(3, stride=2),
        )

        self.training = training

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool3d(3, stride=2)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool3d((2, 3, 2), stride=1)
        self.linear = nn.Linear(27648, 2)

        self.aux1 = InceptionAux(512, 2)
        self.aux2 = InceptionAux(528, 2)

    def forward(self, x, covars = None):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)

        # For gradient injection
        if self.training:
            aux_out1 = self.aux1(out)

        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)

        # For gradient injection
        if self.training:
            aux_out2 = self.aux2(out)

        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = F.dropout(out, 0.4, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.training:
            return out, aux_out1, aux_out2
        return out

class ResBlock(nn.Module):
    def __init__(self, block_number, input_size):
        super(ResBlock, self).__init__()

        layer_in = input_size if input_size is not None else 2 ** (block_number + 1)
        layer_out = 2 ** (block_number + 2)

        self.conv1 = nn.Conv3d(layer_in, layer_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(layer_out)
        self.act1 = nn.ELU()

        self.conv2 = nn.Conv3d(layer_out, layer_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(layer_out)

        # shortcut
        self.shortcut = nn.Sequential(
            nn.Conv3d(layer_in, layer_out, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.act2 = nn.ELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, n_covars=0, input_size=[1, 169, 208, 179], **kwargs):
        super(ResNet, self).__init__()
        assert len(input_size) == 4, "input must be in 3d with the corresponding number of channels"
        self.nb_covars = n_covars

        # self.conv1 = nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm3d(64)

        self.layer1 = self._make_block(1, input_size[0])
        self.layer2 = self._make_block(2)
        self.layer3 = self._make_block(3)
        self.layer4 = self._make_block(4)
        self.layer5 = self._make_block(5)

        d, h, w = ResNet._maxpool_output_size(input_size[1::], nb_layers=5)

        self.fc = nn.Sequential(
            Flatten(),
            # nn.Linear(128*2*3*2, 256),  # wm/gm 128 = 2 ** (5 + 2)  # 5 is for 5 blocks
            # nn.Linear(128*4*5*4, 256),  # t1 image
            nn.Linear(128 * d * h * w, 256),  # t1 image
            nn.ELU(),
            nn.Dropout(p=0.8),
            #AddingNodes(), only for covariables ? 
            nn.Linear(256 + self.nb_covars, 2)
        )

    @staticmethod
    def _make_block(block_number, input_size=None):
        return nn.Sequential(
            ResBlock(block_number, input_size),
            nn.MaxPool3d(3, stride=2)
        )

    @staticmethod
    def _maxpool_output_size(input_size, kernel_size=(3, 3, 3), stride=(2, 2, 2), nb_layers=1):
        import math

        d = math.floor((input_size[0] - kernel_size[0]) / stride[0] + 1)
        h = math.floor((input_size[1] - kernel_size[1]) / stride[1] + 1)
        w = math.floor((input_size[2] - kernel_size[2]) / stride[2] + 1)

        if nb_layers == 1:
            return d, h, w
        return ResNet._maxpool_output_size((d, h, w), kernel_size=kernel_size, stride=stride, nb_layers=nb_layers-1)

    def forward(self, x, covars=None):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        for layer in self.fc:
            print(layer)
            #if isinstance(layer, AddingNodes):
            #    out = layer(out, covars)
            #else:
            out = layer(out)

        # out = self.fc(out)
        return out
