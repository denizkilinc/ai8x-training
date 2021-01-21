###################################################################################################
#
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################

"""
Test networks for AI85/AI86
Optionally quantize/clamp activations
"""

import torch.nn as nn
import ai8x


class AI85SpO2Net(nn.Module):
    """
    Simple MLP Model
    """

    def __init__(
            self,
            num_classes=None,
            num_channels=1,
            dimensions=(32,1),  # pylint: disable=unused-argument
            bias=False,
            **kwargs
    ):

        super().__init__()

        self.linear1 = ai8x.FusedLinearReLU(32, 250, bias=bias, **kwargs)
        self.linear2 = ai8x.FusedLinearReLU(250, 150, bias=bias, **kwargs)
        self.linear3 = ai8x.FusedLinearReLU(150, 70, bias=bias, **kwargs)
        self.linear4 = ai8x.FusedLinearReLU(70, 30, bias=bias, **kwargs)
        self.linear5 = ai8x.FusedLinearReLU(30, 10, bias=bias, **kwargs)
        self.fc = ai8x.Linear(10, 1, bias=bias, wide=True, **kwargs)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)
        

    def forward(self, x):  # pylint: disable=arguments-differ

        x = self.linear1(x)

        x = self.linear2(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        x = self.dropout3(x)

        x = self.linear4(x)

        x = self.linear5(x)

        x = self.fc(x)

        return x



def ai85spo2net(pretrained=False, **kwargs):
    """
    Constructs a MLP model.
    """
    assert not pretrained
    return AI85SpO2Net(**kwargs)


models = [
    {
        'name': 'ai85spo2net',
        'min_input': 1,
        'dim': 1,
    },
]