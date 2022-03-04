import scipy.misc
import scipy.io
from ops import *
from setting import *
import torch
from torchvision import models
from torch import nn


def init_parameters_recursively(layer):
    if isinstance(layer, nn.Sequential):
        for sub_layer in layer:
            init_parameters_recursively(sub_layer)
    elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, std=0.01)
        if layer.bias is not None:
            nn.init.normal_(layer.bias, std=0.01)
    else:
        return


class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        self.cnn = models.vgg11(pretrained=True).type(torch.float32).features
        self.feature = nn.Sequential(
            nn.Linear(512 * 7 * 7, SEMANTIC_EMBED * 2),
        )
        self.hash = nn.Sequential(
            nn.Linear(SEMANTIC_EMBED, bit),
            nn.Tanh()
        )
        self.label = nn.Sequential(
            nn.Linear(SEMANTIC_EMBED, numClass),
            nn.Sigmoid()
        )
        self.cross_feature = nn.Sequential(
            nn.Linear(SEMANTIC_EMBED, SEMANTIC_EMBED),
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.feature)
        init_parameters_recursively(self.hash)
        init_parameters_recursively(self.label)
        init_parameters_recursively(self.cross_feature)

    def forward(self, inputs):
        base = self.cnn(inputs).view(inputs.shape[0], -1)
        mu_sigma_I = self.feature(base)
        mu_I = mu_sigma_I[:, :SEMANTIC_EMBED]
        log_sigma_I = mu_sigma_I[:, SEMANTIC_EMBED:]
        std_I = log_sigma_I.mul(0.5).exp_()
        fea_I = torch.relu(torch.randn_like(mu_I) * std_I + mu_I)
        fea_T_pred = self.cross_feature(fea_I)
        hsh_I = self.hash(fea_I)
        lab_I = self.label(fea_I)
        return torch.squeeze(fea_I), torch.squeeze(hsh_I), torch.squeeze(lab_I), fea_T_pred, mu_I, log_sigma_I


class LabelNet(nn.Module):
    def __init__(self):
        super(LabelNet, self).__init__()
        self.hash = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=bit, kernel_size=(1, numClass), stride=(1, 1), bias=False),
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.hash)

    def forward(self, inputs):
        hsh_I = self.hash(inputs.view(inputs.shape[0], 1, 1, -1))
        return torch.squeeze(hsh_I)


class TextNet(nn.Module):
    def __init__(self):
        super(TextNet, self).__init__()
        self.interp_block1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[1, 1 * 5], stride=[1, 1 * 5]),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )
        self.interp_block2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[1, 2 * 5], stride=[1, 2 * 5]),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )
        self.interp_block3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[1, 3 * 5], stride=[1, 3 * 5]),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )
        self.interp_block6 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[1, 6 * 5], stride=[1, 6 * 5]),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )
        self.interp_block10 = nn.Sequential(
            nn.AvgPool2d(kernel_size=[1, 10 * 5], stride=[1, 10 * 5]),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=4096, kernel_size=(1, dimTxt), stride=(1, 1)),
            nn.ReLU(),
            nn.modules.normalization.LocalResponseNorm(size=4, alpha=0.0001, beta=0.75, k=2.0),
            nn.Conv2d(in_channels=4096, out_channels=SEMANTIC_EMBED, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
        )
        self.norm = nn.modules.normalization.LocalResponseNorm(size=4, alpha=0.0001, beta=0.75, k=2.0)
        self.hash = nn.Sequential(
            nn.Conv2d(in_channels=SEMANTIC_EMBED, out_channels=bit, kernel_size=(1, 1), stride=(1, 1)),
            nn.Tanh(),
        )
        self.label = nn.Sequential(
            nn.Conv2d(in_channels=SEMANTIC_EMBED, out_channels=numClass, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
        self.init_parameters()

    def init_parameters(self):
        init_parameters_recursively(self.interp_block1)
        init_parameters_recursively(self.interp_block2)
        init_parameters_recursively(self.interp_block3)
        init_parameters_recursively(self.interp_block6)
        init_parameters_recursively(self.interp_block10)
        init_parameters_recursively(self.feature)
        init_parameters_recursively(self.hash)
        init_parameters_recursively(self.label)

    def forward(self, inputs):
        unsqueezed = inputs.view(inputs.shape[0], 1, 1, -1)
        interp_in1 = nn.functional.interpolate(self.interp_block1(unsqueezed), size=(1, dimTxt))
        interp_in2 = nn.functional.interpolate(self.interp_block2(unsqueezed), size=(1, dimTxt))
        interp_in3 = nn.functional.interpolate(self.interp_block3(unsqueezed), size=(1, dimTxt))
        interp_in6 = nn.functional.interpolate(self.interp_block6(unsqueezed), size=(1, dimTxt))
        interp_in10 = nn.functional.interpolate(self.interp_block10(unsqueezed), size=(1, dimTxt))
        MultiScal = torch.cat([
            unsqueezed,
            interp_in10,
            interp_in6,
            interp_in3,
            interp_in2,
            interp_in1
        ], 1)
        feature = self.feature(MultiScal)
        norm = self.norm(feature)
        hash = self.hash(norm)
        label = self.label(norm)

        return torch.squeeze(feature), torch.squeeze(hash), torch.squeeze(label)
