import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR

from collections import OrderedDict
from layers import *
import cv2

class Encoder_Decoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Encoder_Decoder, self).__init__()

        self.encoder = encoder

        self.depth_decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.depth_decoder(features)
        return outputs[("disp", 0)]

