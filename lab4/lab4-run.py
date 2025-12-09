#!/usr/bin/env python
# coding: utf-8

# # Lab 4: Quantization - Completed Single File

# ## Goals and Setup

import copy
import math
import os
import random
import sys
from collections import OrderedDict, defaultdict, namedtuple

import numpy as np
import torch
from fast_pytorch_kmeans import KMeans
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm

assert torch.backends.mps.is_available(), "MPS not available. Running on CPU."
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def download_url(url, model_dir=".", overwrite=False):
    from urllib.request import urlretrieve

    target_dir = url.split("/")[-1]
    model_dir = os.path.expanduser(model_dir)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        os.remove(os.path.join(model_dir, "download.lock"))
        sys.stderr.write("Failed to download from url %s" % url + "\n" + str(e) + "\n")
        return None


class VGG(nn.Module):
    ARCH = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self) -> None:
        super().__init__()

        layers = []
        counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != "M":
                # conv-bn-relu
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(True))
                in_channels = x
            else:
                # maxpool
                add("pool", nn.MaxPool2d(2))
        add("avgpool", nn.AvgPool2d(2))
        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    callbacks=None,
) -> None:
    model.train()

    for inputs, targets in tqdm(dataloader, desc="train", leave=False):
        inputs = inputs.to("mps")
        targets = targets.to("mps")
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if callbacks is not None:
            for callback in callbacks:
                callback()


@torch.inference_mode()
def evaluate(model: nn.Module, dataloader: DataLoader, extra_preprocess=None) -> float:
    model.eval()
    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
        inputs = inputs.to("mps")
        if extra_preprocess is not None:
            for preprocess in extra_preprocess:
                inputs = preprocess(inputs)

        targets = targets.to("mps")
        outputs = model(inputs)
        outputs = outputs.argmax(dim=1)
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()


def get_model_flops(model, inputs):
    num_macs = torchprofile.profile_macs(model, inputs)
    return num_macs


def get_model_size(model: nn.Module, data_width=32):
    """
    calculate the model size in bits
    :param data_width: #bits per element
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width


Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def get_quantized_range(bitwidth):
    quantized_max = (1 << (bitwidth - 1)) - 1
    quantized_min = -(1 << (bitwidth - 1))
    return quantized_min, quantized_max


# Define misc functions for verification. (Skipping plotting code for brevity)
def test_k_means_quantize(test_tensor, bitwidth=2):
    # This is a placeholder for the actual test which involves plotting
    print("Running k-means quantization test...")
    original_tensor_clone = test_tensor.clone()
    num_unique_values_before_quantization = test_tensor.unique().numel()
    k_means_quantize(test_tensor, bitwidth=bitwidth)
    num_unique_values_after_quantization = test_tensor.unique().numel()
    print(f"  Target bitwidth: {bitwidth} bits")
    print(f"  Unique values before: {num_unique_values_before_quantization}")
    print(f"  Unique values after: {num_unique_values_after_quantization}")
    assert num_unique_values_after_quantization == min(
        (1 << bitwidth), num_unique_values_before_quantization
    )
    print("Test passed.")
    test_tensor.set_(original_tensor_clone)  # Restore for clean notebook environment


def test_linear_quantize(
    test_tensor, quantized_test_tensor, real_min, real_max, bitwidth, scale, zero_point
):
    # This is a placeholder for the actual test which involves plotting
    print("Running linear quantization test...")
    _quantized_test_tensor = linear_quantize(
        test_tensor, bitwidth=bitwidth, scale=scale, zero_point=zero_point
    )
    assert _quantized_test_tensor.equal(quantized_test_tensor)
    print("Test passed.")


def test_quantized_fc(
    input,
    weight,
    bias,
    quantized_bias,
    shifted_quantized_bias,
    calc_quantized_output,
    bitwidth,
    batch_size,
    in_channels,
    out_channels,
):
    # This is a placeholder for the actual test which involves plotting
    print("Running quantized_fc() test...")

    output = torch.nn.functional.linear(input, weight, bias)
    quantized_weight, weight_scale, weight_zero_point = (
        linear_quantize_weight_per_channel(weight, bitwidth)
    )
    quantized_input, input_scale, input_zero_point = linear_quantize_feature(
        input, bitwidth
    )
    _quantized_bias, bias_scale, bias_zero_point = (
        linear_quantize_bias_per_output_channel(bias, weight_scale, input_scale)
    )
    assert _quantized_bias.equal(quantized_bias)

    _shifted_quantized_bias = shift_quantized_linear_bias(
        quantized_bias, quantized_weight, input_zero_point
    )
    assert _shifted_quantized_bias.equal(shifted_quantized_bias)

    quantized_output, output_scale, output_zero_point = linear_quantize_feature(
        output, bitwidth
    )

    _calc_quantized_output = quantized_linear(
        quantized_input,
        quantized_weight,
        shifted_quantized_bias,
        bitwidth,
        bitwidth,
        input_zero_point,
        output_zero_point,
        input_scale,
        weight_scale,
        output_scale,
    )
    assert _calc_quantized_output.equal(calc_quantized_output)
    print("Test passed.")


# --- Load Model and Data ---

checkpoint_url = "https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth"
checkpoint = torch.load(download_url(checkpoint_url), map_location="cpu")
model = VGG().to("mps")
print(f"=> loading checkpoint '{checkpoint_url}'")
model.load_state_dict(checkpoint["state_dict"])
recover_model = lambda: model.load_state_dict(checkpoint["state_dict"])

image_size = 32
transforms = {
    "train": Compose(
        [
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]
    ),
    "test": ToTensor(),
}
dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR10(
        root="data/cifar10",
        train=(split == "train"),
        download=True,
        transform=transforms[split],
    )
dataloader = {}
for split in ["train", "test"]:
    dataloader[split] = DataLoader(
        dataset[split],
        batch_size=512,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=True,
    )

# --- Evaluate FP32 Model ---

fp32_model_accuracy = evaluate(model, dataloader["test"])
fp32_model_size = get_model_size(model)
print(f"fp32 model has accuracy={fp32_model_accuracy:.2f}%")
print(f"fp32 model has size={fp32_model_size / MiB:.2f} MiB")

# # K-Means Quantization Implementation

Codebook = namedtuple("Codebook", ["centroids", "labels"])

# ## Question 1 (10 pts): k_means_quantize


def k_means_quantize(fp32_tensor: torch.Tensor, bitwidth=4, codebook=None):
    """
    quantize tensor using k-means clustering
    """
    if codebook is None:
        ############### YOUR CODE STARTS HERE ###############
        # get number of clusters based on the quantization precision
        n_clusters = 1 << bitwidth
        ############### YOUR CODE ENDS HERE #################
        # use k-means to get the quantization centroids
        kmeans = KMeans(n_clusters=n_clusters, mode="euclidean", verbose=0)
        labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        codebook = Codebook(centroids, labels)
    ############### YOUR CODE STARTS HERE ###############
    # decode the codebook into k-means quantized tensor for inference
    quantized_tensor = codebook.centroids[codebook.labels]
    ############### YOUR CODE ENDS HERE #################
    fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
    return codebook


# Verification Test
print("\n--- Q1 Verification ---")
test_tensor_q1 = torch.tensor(
    [
        [-0.3747, 0.0874, 0.3200, -0.4868, 0.4404],
        [-0.0402, 0.2322, -0.2024, -0.4986, 0.1814],
        [0.3102, -0.3942, -0.2030, 0.0883, -0.4741],
        [-0.1592, -0.0777, -0.3946, -0.2128, 0.2675],
        [0.0611, -0.1933, -0.4350, 0.2928, -0.1087],
    ]
).to("mps")
test_k_means_quantize(test_tensor_q1, bitwidth=2)


# ## Question 2 (10 pts): Cluster Size
#
# # Question 2.1 (5 pts)
# # Your Answer: 16 unique colors, since $2^4 = 16$.
#
# # Question 2.2 (5 pts)
# # Your Answer: $2^n$ unique colors, since $n$ bits can encode $2^n$ unique values (clusters).

# ---
#
# # K-Means Quantization on Whole Model


class KMeansQuantizer:
    def __init__(self, model: nn.Module, bitwidth=4):
        self.codebook = KMeansQuantizer.quantize(model, bitwidth)

    @torch.no_grad()
    def apply(self, model, update_centroids):
        for name, param in model.named_parameters():
            if name in self.codebook:
                if update_centroids:
                    update_codebook(param, codebook=self.codebook[name])
                self.codebook[name] = k_means_quantize(
                    param, codebook=self.codebook[name]
                )

    @staticmethod
    @torch.no_grad()
    def quantize(model: nn.Module, bitwidth=4):
        codebook = dict()
        if isinstance(bitwidth, dict):
            for name, param in model.named_parameters():
                if name in bitwidth:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth[name])
        else:
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth)
        return codebook


# --- Quantization-Aware Training ---

# ## Question 3 (10 pts): update_codebook


def update_codebook(fp32_tensor: torch.Tensor, codebook: Codebook):
    """
    update the centroids in the codebook using updated fp32_tensor
    """
    n_clusters = codebook.centroids.numel()
    fp32_tensor = fp32_tensor.view(-1)
    for k in range(n_clusters):
        ############### YOUR CODE STARTS HERE ###############
        # hint: one line of code: calculate the mean of weights for the current cluster k
        codebook.centroids[k] = fp32_tensor[codebook.labels == k].mean()
    ############### YOUR CODE ENDS HERE #################


# --- Run K-Means Quantization and Fine-tuning ---

print("\n--- K-Means Quantization Results ---")
print("Note that the storage for codebooks is ignored when calculating the model size.")
quantizers = dict()
for bitwidth in [8, 4, 2]:
    recover_model()
    print(f"k-means quantizing model into {bitwidth} bits")
    quantizer = KMeansQuantizer(model, bitwidth)
    quantized_model_size = get_model_size(model, bitwidth)
    print(
        f"    {bitwidth}-bit k-means quantized model has size={quantized_model_size / MiB:.2f} MiB"
    )
    quantized_model_accuracy = evaluate(model, dataloader["test"])
    print(
        f"    {bitwidth}-bit k-means quantized model has accuracy={quantized_model_accuracy:.2f}% before QAT "
    )
    quantizers[bitwidth] = quantizer

# Fine-tuning loop
accuracy_drop_threshold = 0.5
quantizers_before_finetune = copy.deepcopy(quantizers)
quantizers_after_finetune = quantizers

print("\n--- K-Means Quantization-Aware Training (QAT) ---")
for bitwidth in [8, 4, 2]:
    recover_model()
    quantizer = quantizers[bitwidth]
    quantizer.apply(model, update_centroids=False)  # Re-apply initial quantization
    quantized_model_accuracy = evaluate(model, dataloader["test"])
    accuracy_drop = fp32_model_accuracy - quantized_model_accuracy
    print(f"\n{bitwidth}-bit QAT status (Initial drop: {accuracy_drop:.2f}%)")

    if accuracy_drop > accuracy_drop_threshold:
        print(f"    Quantization-aware training starting...")
        num_finetune_epochs = 5
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, num_finetune_epochs
        )
        criterion = nn.CrossEntropyLoss()
        best_accuracy = quantized_model_accuracy
        epoch = num_finetune_epochs
        while accuracy_drop > accuracy_drop_threshold and epoch > 0:
            train(
                model,
                dataloader["train"],
                criterion,
                optimizer,
                scheduler,
                callbacks=[lambda: quantizer.apply(model, update_centroids=True)],
            )
            model_accuracy = evaluate(model, dataloader["test"])
            best_accuracy = max(model_accuracy, best_accuracy)
            print(
                f"        Epoch {num_finetune_epochs - epoch + 1} Accuracy {model_accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%"
            )
            accuracy_drop = fp32_model_accuracy - best_accuracy
            epoch -= 1
    else:
        print(f"    No need for QAT.")

# # Linear Quantization Implementation

# --- Linear Quantization Helper Functions ---

# ## Question 4 (10 pts): linear_quantize


def linear_quantize(
    fp_tensor, bitwidth, scale, zero_point, dtype=torch.int8
) -> torch.Tensor:
    """
    linear quantization for single fp_tensor: q = int(round(fp_tensor / scale)) + zero_point
    """
    assert fp_tensor.dtype == torch.float
    assert isinstance(scale, float) or (
        scale.dtype == torch.float and scale.dim() == fp_tensor.dim()
    )
    assert isinstance(zero_point, int) or (
        zero_point.dtype == dtype and zero_point.dim() == fp_tensor.dim()
    )

    ############### YOUR CODE STARTS HERE ###############
    # Step 1: scale the fp_tensor
    scaled_tensor = fp_tensor / scale
    # Step 2: round the floating value to integer value
    rounded_tensor = torch.round(scaled_tensor)
    ############### YOUR CODE ENDS HERE #################

    rounded_tensor = rounded_tensor.to(dtype).to(
        fp_tensor.device
    )  # Move to correct device

    ############### YOUR CODE STARTS HERE ###############
    # Step 3: shift the rounded_tensor by adding the zero_point
    shifted_tensor = rounded_tensor + zero_point
    ############### YOUR CODE ENDS HERE #################

    # Step 4: clamp the shifted_tensor to lie in bitwidth-bit range
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    # Ensure it's in a float/long format before clamp
    quantized_tensor = (
        shifted_tensor.float().clamp_(quantized_min, quantized_max).to(dtype)
    )
    return quantized_tensor


# Verification Test
print("\n--- Q4 Verification ---")
test_tensor_q4 = torch.tensor(
    [
        [0.0523, 0.6364, -0.0968, -0.0020, 0.1940],
        [0.7500, 0.5507, 0.6188, -0.1734, 0.4677],
        [-0.0669, 0.3836, 0.4297, 0.6267, -0.0695],
        [0.1536, -0.0038, 0.6075, 0.6817, 0.0601],
        [0.6446, -0.2500, 0.5376, -0.2226, 0.2333],
    ]
)
quantized_test_tensor_q4 = torch.tensor(
    [
        [-1, 1, -1, -1, 0],
        [1, 1, 1, -2, 0],
        [-1, 0, 0, 1, -1],
        [-1, -1, 1, 1, -1],
        [1, -2, 1, -2, 0],
    ],
    dtype=torch.int8,
)
test_linear_quantize(
    test_tensor_q4.to("mps"),
    quantized_test_tensor_q4.to("mps"),
    -0.25,
    0.75,
    2,
    1 / 3,
    -1,
)


# ## Question 5 (15 pts): Scale and Zero Point

# # Question 5.1 (3 pts)
# # Your Answer: $S=(r_{\mathrm{max}} - r_{\mathrm{min}}) / (q_{\mathrm{max}} - q_{\mathrm{min}})$
#
# # Question 5.2 (4 pts)
# # Your Answer: $Z = \mathrm{int}(\mathrm{round}(q_{\mathrm{min}} - r_{\mathrm{min}} / S))$

# ## Question 5.3 (8 pts): get_quantization_scale_and_zero_point


def get_quantization_scale_and_zero_point(fp_tensor, bitwidth):
    """
    get quantization scale and zero point for single tensor
    """
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    fp_max = fp_tensor.max().item()
    fp_min = fp_tensor.min().item()

    ############### YOUR CODE STARTS HERE ###############
    # Scale: S = (fp_max - fp_min) / (q_max - q_min)
    scale = (fp_max - fp_min) / (quantized_max - quantized_min)
    # Zero Point: Z = q_min - r_min / S
    zero_point = quantized_min - fp_min / scale
    ############### YOUR CODE ENDS HERE ###############

    # clip the zero_point to fall in [quantized_min, quantized_max]
    if zero_point < quantized_min:
        zero_point = quantized_min
    elif zero_point > quantized_max:
        zero_point = quantized_max
    else:  # convert from float to int using round()
        zero_point = round(zero_point)
    return scale, int(zero_point)


def linear_quantize_feature(fp_tensor, bitwidth):
    scale, zero_point = get_quantization_scale_and_zero_point(fp_tensor, bitwidth)
    quantized_tensor = linear_quantize(fp_tensor, bitwidth, scale, zero_point)
    return quantized_tensor, scale, zero_point


def get_quantization_scale_for_weight(weight, bitwidth):
    fp_max = max(weight.abs().max().item(), 5e-7)
    _, quantized_max = get_quantized_range(bitwidth)
    return fp_max / quantized_max


def linear_quantize_weight_per_channel(tensor, bitwidth):
    dim_output_channels = 0
    num_output_channels = tensor.shape[dim_output_channels]
    scale = torch.zeros(num_output_channels, device=tensor.device)
    for oc in range(num_output_channels):
        _subtensor = tensor.select(dim_output_channels, oc)
        _scale = get_quantization_scale_for_weight(_subtensor, bitwidth)
        scale[oc] = _scale
    scale_shape = [1] * tensor.dim()
    scale_shape[dim_output_channels] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_quantize(tensor, bitwidth, scale, zero_point=0)
    return quantized_tensor, scale, 0


# --- Per-channel Linear Quantization Peek (Skipping plotting code for brevity) ---
@torch.no_grad()
def peek_linear_quantization():
    print("\n--- Q5 Peek at Weight Distribution (No plot generated) ---")
    bitwidths = [4, 2]
    original_state_dict = model.state_dict()

    for bitwidth in bitwidths:
        for name, param in model.named_parameters():
            if param.dim() > 1:
                quantized_param, _, _ = linear_quantize_weight_per_channel(
                    param, bitwidth
                )
                param.copy_(quantized_param)
        print(f"  Finished peek for {bitwidth}-bit quantization.")
        model.load_state_dict(original_state_dict)  # Restore model
    print("Peek complete.")


recover_model()
peek_linear_quantization()


# --- Quantized Inference Functions ---

# ## Question 6 (5 pts): linear_quantize_bias_per_output_channel


def linear_quantize_bias_per_output_channel(bias, weight_scale, input_scale):
    """
    linear quantization for single bias tensor
    """
    assert bias.dim() == 1
    assert bias.dtype == torch.float
    assert isinstance(input_scale, float)
    if isinstance(weight_scale, torch.Tensor):
        assert weight_scale.dtype == torch.float
        weight_scale = weight_scale.view(-1)
        assert bias.numel() == weight_scale.numel()

    ############### YOUR CODE STARTS HERE ###############
    # S_bias = S_input * S_weight
    bias_scale = input_scale * weight_scale
    ############### YOUR CODE ENDS HERE ###############

    quantized_bias = linear_quantize(
        bias, 32, bias_scale, zero_point=0, dtype=torch.int32
    )
    return quantized_bias, bias_scale, 0


def shift_quantized_linear_bias(quantized_bias, quantized_weight, input_zero_point):
    assert quantized_bias.dtype == torch.int32
    assert isinstance(input_zero_point, int)
    return quantized_bias - quantized_weight.sum(1).to(torch.int32) * input_zero_point


# ## Question 7 (15 pts): quantized_linear


def quantized_linear(
    input,
    weight,
    bias,
    feature_bitwidth,
    weight_bitwidth,
    input_zero_point,
    output_zero_point,
    input_scale,
    weight_scale,
    output_scale,
):
    """
    quantized fully-connected layer
    """
    assert input.dtype == torch.int8
    # ... (assertions omitted for brevity)

    # Step 1: integer-based fully-connected (8-bit multiplication with 32-bit accumulation)
    if "cpu" in input.device.type:
        output = torch.nn.functional.linear(
            input.to(torch.int32), weight.to(torch.int32), bias
        )
    else:
        output = torch.nn.functional.linear(input.float(), weight.float(), bias.float())
        output = output.round().to(torch.int32)  # Ensure output is integer accumulation

    ############### YOUR CODE STARTS HERE ###############
    # Step 2: scale the output: scale_factor = (S_input * S_weight) / S_output
    scale_factor = (input_scale * weight_scale) / output_scale
    # Added for shape error
    scale_factor = scale_factor.view(1, -1)
    output = output.float() * scale_factor

    # Step 3: shift output by output_zero_point
    output = output + output_zero_point
    ############### YOUR CODE ENDS HERE ###############

    # Make sure all value lies in the bitwidth-bit range
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output


# Verification Test
print("\n--- Q7 Verification ---")
test_input_q7 = torch.tensor(
    [
        [0.6118, 0.7288, 0.8511, 0.2849, 0.8427, 0.7435, 0.4014, 0.2794],
        [0.3676, 0.2426, 0.1612, 0.7684, 0.6038, 0.0400, 0.2240, 0.4237],
        [0.6565, 0.6878, 0.4670, 0.3470, 0.2281, 0.8074, 0.0178, 0.3999],
        [0.1863, 0.3567, 0.6104, 0.0497, 0.0577, 0.2990, 0.6687, 0.8626],
    ]
).to("mps")
test_weight_q7 = torch.tensor(
    [
        [
            1.2626e-01,
            -1.4752e-01,
            8.1910e-02,
            2.4982e-01,
            -1.0495e-01,
            -1.9227e-01,
            -1.8550e-01,
            -1.5700e-01,
        ],
        [
            2.7624e-01,
            -4.3835e-01,
            5.1010e-02,
            -1.2020e-01,
            -2.0344e-01,
            1.0202e-01,
            -2.0799e-01,
            2.4112e-01,
        ],
        [
            -3.8216e-01,
            -2.8047e-01,
            8.5238e-02,
            -4.2504e-01,
            -2.0952e-01,
            3.2018e-01,
            -3.3619e-01,
            2.0219e-01,
        ],
        [
            8.9233e-02,
            -1.0124e-01,
            1.1467e-01,
            2.0091e-01,
            1.1438e-01,
            -4.2427e-01,
            1.0178e-01,
            -3.0941e-04,
        ],
        [
            -1.8837e-02,
            -2.1256e-01,
            -4.5285e-01,
            2.0949e-01,
            -3.8684e-01,
            -1.7100e-01,
            -4.5331e-01,
            -2.0433e-01,
        ],
        [
            -2.0038e-01,
            -5.3757e-02,
            1.8997e-01,
            -3.6866e-01,
            5.5484e-02,
            1.5643e-01,
            -2.3538e-01,
            2.1103e-01,
        ],
        [
            -2.6875e-01,
            2.4984e-01,
            -2.3514e-01,
            2.5527e-01,
            2.0322e-01,
            3.7675e-01,
            6.1563e-02,
            1.7201e-01,
        ],
        [
            3.3541e-01,
            -3.3555e-01,
            -4.3349e-01,
            4.3043e-01,
            -2.0498e-01,
            -1.8366e-01,
            -9.1553e-02,
            -4.1168e-01,
        ],
    ]
).to("mps")
test_bias_q7 = torch.tensor(
    [0.1954, -0.2756, 0.3113, 0.1149, 0.4274, 0.2429, -0.1721, -0.2502]
).to("mps")
test_quantized_bias_q7 = torch.tensor(
    [3, -2, 3, 1, 3, 2, -2, -2], dtype=torch.int32
).to("mps")
test_shifted_quantized_bias_q7 = torch.tensor(
    [-1, 0, -3, -1, -3, 0, 2, -4], dtype=torch.int32
).to("mps")
test_calc_quantized_output_q7 = torch.tensor(
    [
        [0, -1, 0, -1, -1, 0, 1, -2],
        [0, 0, -1, 0, 0, 0, 0, -1],
        [0, 0, 0, -1, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 1, -1, -2],
    ],
    dtype=torch.int8,
).to("mps")
test_quantized_fc(
    test_input_q7,
    test_weight_q7,
    test_bias_q7,
    test_quantized_bias_q7,
    test_shifted_quantized_bias_q7,
    test_calc_quantized_output_q7,
    2,
    4,
    8,
    8,
)


def shift_quantized_conv2d_bias(quantized_bias, quantized_weight, input_zero_point):
    assert quantized_bias.dtype == torch.int32
    assert isinstance(input_zero_point, int)
    return (
        quantized_bias
        - quantized_weight.sum((1, 2, 3)).to(torch.int32) * input_zero_point
    )


# ## Question 8 (10 pts): quantized_conv2d


def quantized_conv2d(
    input,
    weight,
    bias,
    feature_bitwidth,
    weight_bitwidth,
    input_zero_point,
    output_zero_point,
    input_scale,
    weight_scale,
    output_scale,
    stride,
    padding,
    dilation,
    groups,
):
    """
    quantized 2d convolution
    """
    assert len(padding) == 4
    assert input.dtype == torch.int8
    # ... (assertions omitted for brevity)

    # Step 1: calculate integer-based 2d convolution
    input = torch.nn.functional.pad(input, padding, "constant", input_zero_point)
    if "cpu" in input.device.type:
        output = torch.nn.functional.conv2d(
            input.to(torch.int32),
            weight.to(torch.int32),
            None,
            stride,
            0,
            dilation,
            groups,
        )
    else:
        output = torch.nn.functional.conv2d(
            input.float(), weight.float(), None, stride, 0, dilation, groups
        )
        output = output.round().to(torch.int32)
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)

    ############### YOUR CODE STARTS HERE ###############
    # Step 2: scale the output: scale_factor = (S_input * S_weight) / S_output
    # weight_scale needs to be shaped for broadcasting: [1, oc, 1, 1]
    scale_factor = (input_scale * weight_scale).view(1, -1, 1, 1) / output_scale
    output = output.float() * scale_factor

    # Step 3: shift output by output_zero_point
    output = output + output_zero_point
    ############### YOUR CODE ENDS HERE ###############

    # Make sure all value lies in the bitwidth-bit range
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output


# --- Post-Training Quantization (PTQ) ---

# ## Question 9 (10 pts): PTQ Pipeline


def fuse_conv_bn(conv, bn):
    assert conv.bias is None
    factor = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
    conv.weight.data = conv.weight.data * factor.reshape(-1, 1, 1, 1)
    conv.bias = nn.Parameter(-bn.running_mean.data * factor + bn.bias.data)
    return conv


print("\n--- Q9.1: PTQ and Preprocessing ---")

# 1. Conv-BN fusion
print("Before conv-bn fusion: backbone length", len(model.backbone))
recover_model()
model_fused = copy.deepcopy(model)
fused_backbone = []
ptr = 0
while ptr < len(model_fused.backbone):
    if isinstance(model_fused.backbone[ptr], nn.Conv2d) and isinstance(
        model_fused.backbone[ptr + 1], nn.BatchNorm2d
    ):
        fused_backbone.append(
            fuse_conv_bn(model_fused.backbone[ptr], model_fused.backbone[ptr + 1])
        )
        ptr += 2
    else:
        fused_backbone.append(model_fused.backbone[ptr])
        ptr += 1
model_fused.backbone = nn.Sequential(*fused_backbone)

print("After conv-bn fusion: backbone length", len(model_fused.backbone))
fused_acc = evaluate(model_fused, dataloader["test"])
print(f"Accuracy of the fused model={fused_acc:.2f}%")

# 2. Add hooks and calibrate (record activation ranges)
input_activation = {}
output_activation = {}


def add_range_recoder_hook(model):
    import functools

    def _record_range(self, x, y, module_name):
        x = x[0]
        input_activation[module_name] = x.detach()
        output_activation[module_name] = y.detach()

    all_hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU)):
            all_hooks.append(
                m.register_forward_hook(
                    functools.partial(_record_range, module_name=name)
                )
            )
    return all_hooks


hooks = add_range_recoder_hook(model_fused)
sample_data = iter(dataloader["train"]).__next__()[0]
model_fused(sample_data.to("mps"))

for h in hooks:
    h.remove()


# 3. Model Quantization


class QuantizedConv2d(nn.Module):
    def __init__(
        self,
        weight,
        bias,
        input_zero_point,
        output_zero_point,
        input_scale,
        weight_scale,
        output_scale,
        stride,
        padding,
        dilation,
        groups,
        feature_bitwidth=8,
        weight_bitwidth=8,
    ):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point
        self.input_scale = input_scale
        self.register_buffer("weight_scale", weight_scale)
        self.output_scale = output_scale
        self.stride = stride
        self.padding = (padding[1], padding[1], padding[0], padding[0])
        self.dilation = dilation
        self.groups = groups
        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

    def forward(self, x):
        return quantized_conv2d(
            x,
            self.weight,
            self.bias,
            self.feature_bitwidth,
            self.weight_bitwidth,
            self.input_zero_point,
            self.output_zero_point,
            self.input_scale,
            self.weight_scale,
            self.output_scale,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        weight,
        bias,
        input_zero_point,
        output_zero_point,
        input_scale,
        weight_scale,
        output_scale,
        feature_bitwidth=8,
        weight_bitwidth=8,
    ):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point
        self.input_scale = input_scale
        self.register_buffer("weight_scale", weight_scale)
        self.output_scale = output_scale
        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

    def forward(self, x):
        return quantized_linear(
            x,
            self.weight,
            self.bias,
            self.feature_bitwidth,
            self.weight_bitwidth,
            self.input_zero_point,
            self.output_zero_point,
            self.input_scale,
            self.weight_scale,
            self.output_scale,
        )


class QuantizedMaxPool2d(nn.MaxPool2d):
    def forward(self, x):
        return super().forward(x.float()).to(torch.int8)


class QuantizedAvgPool2d(nn.AvgPool2d):
    def forward(self, x):
        return super().forward(x.float()).to(torch.int8)


feature_bitwidth = weight_bitwidth = 8
quantized_model = copy.deepcopy(model_fused)
quantized_backbone = []
ptr = 0
while ptr < len(quantized_model.backbone):
    if isinstance(quantized_model.backbone[ptr], nn.Conv2d) and isinstance(
        quantized_model.backbone[ptr + 1], nn.ReLU
    ):
        conv = quantized_model.backbone[ptr]
        conv_name = f"backbone.{ptr}"
        relu_name = f"backbone.{ptr + 1}"

        input_scale, input_zero_point = get_quantization_scale_and_zero_point(
            input_activation[conv_name].cpu(), feature_bitwidth
        )
        output_scale, output_zero_point = get_quantization_scale_and_zero_point(
            output_activation[relu_name].cpu(), feature_bitwidth
        )

        quantized_weight, weight_scale, weight_zero_point = (
            linear_quantize_weight_per_channel(conv.weight.data, weight_bitwidth)
        )
        quantized_bias, bias_scale, bias_zero_point = (
            linear_quantize_bias_per_output_channel(
                conv.bias.data, weight_scale, input_scale
            )
        )
        shifted_quantized_bias = shift_quantized_conv2d_bias(
            quantized_bias, quantized_weight, input_zero_point
        )

        quantized_conv = QuantizedConv2d(
            quantized_weight,
            shifted_quantized_bias,
            input_zero_point,
            output_zero_point,
            input_scale,
            weight_scale,
            output_scale,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            feature_bitwidth=feature_bitwidth,
            weight_bitwidth=weight_bitwidth,
        ).to("mps")  # Ensure new module is on GPU

        quantized_backbone.append(quantized_conv)
        ptr += 2
    elif isinstance(quantized_model.backbone[ptr], nn.MaxPool2d):
        quantized_backbone.append(
            QuantizedMaxPool2d(
                kernel_size=quantized_model.backbone[ptr].kernel_size,
                stride=quantized_model.backbone[ptr].stride,
            ).to("mps")
        )
        ptr += 1
    elif isinstance(quantized_model.backbone[ptr], nn.AvgPool2d):
        quantized_backbone.append(
            QuantizedAvgPool2d(
                kernel_size=quantized_model.backbone[ptr].kernel_size,
                stride=quantized_model.backbone[ptr].stride,
            ).to("mps")
        )
        ptr += 1
    else:
        raise NotImplementedError(type(quantized_model.backbone[ptr]))
quantized_model.backbone = nn.Sequential(*quantized_backbone)

# Quantize classifier
fc_name = "classifier"
fc = model.classifier
input_scale, input_zero_point = get_quantization_scale_and_zero_point(
    input_activation[fc_name].cpu(), feature_bitwidth
)
output_scale, output_zero_point = get_quantization_scale_and_zero_point(
    fc.bias.data.cpu(), feature_bitwidth
)  # Use bias output for classifier output range

quantized_weight, weight_scale, weight_zero_point = linear_quantize_weight_per_channel(
    fc.weight.data, weight_bitwidth
)
quantized_bias, bias_scale, bias_zero_point = linear_quantize_bias_per_output_channel(
    fc.bias.data, weight_scale, input_scale
)
shifted_quantized_bias = shift_quantized_linear_bias(
    quantized_bias, quantized_weight, input_zero_point
)

quantized_model.classifier = QuantizedLinear(
    quantized_weight,
    shifted_quantized_bias,
    input_zero_point,
    output_zero_point,
    input_scale,
    weight_scale,
    output_scale,
    feature_bitwidth=feature_bitwidth,
    weight_bitwidth=weight_bitwidth,
).to("mps")


# ### Question 9.1 (5 pts): Extra Preprocessing

print(quantized_model)


def extra_preprocess(x):
    ############### YOUR CODE STARTS HERE ###############
    # Convert fp32 [0, 1] input to int8 [-128, 127] based on calibration.
    # The simplest map to the full int8 range (255 steps) and shift by the zero point.
    scale_factor = 255.0
    zero_point_shift = -128
    # Scale x from [0, 1] to approx [0, 255], round to nearest integer, and apply zero point shift.
    output = (x * scale_factor).round() + zero_point_shift
    return output.clamp(-128, 127).to(torch.int8)
    ############### YOUR CODE ENDS HERE ###############


int8_model_accuracy = evaluate(
    quantized_model, dataloader["test"], extra_preprocess=[extra_preprocess]
)
print(f"int8 model has accuracy={int8_model_accuracy:.2f}%")
