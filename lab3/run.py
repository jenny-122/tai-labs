import copy
import math
import random
import time
from collections import OrderedDict, defaultdict
from typing import List, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm

assert torch.backends.mps.is_available(), "MPS not available. Running on CPU."
