from collections import OrderedDict
from matplotlib.pyplot import isinteractive
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


def prune_model(input_model, prune_end):
    pruned_layers = OrderedDict()

    # Loop through named children and prune up to the specified index
    for name, layer in input_model.named_children():
        pruned_layers[name] = layer
        if (isinstance(prune_end, int) and (len(pruned_layers) == prune_end)) or name == prune_end:
            break

    # Create a new sequential model with the pruned layers
    pruned_model = nn.Sequential(pruned_layers)
    return pruned_model


def duplicate_weights(weights, target_depth, axis=0):
    n = weights.shape[axis]
    repetitions, remainder = divmod(target_depth, n)
    subset = [torch.narrow(weights, axis, 0, remainder)] if remainder else []
    duplicated_weights = torch.concat([weights]*repetitions + subset, dim=axis)
    return duplicated_weights


def make_multispectral_resnet(n, input_dim=30, output_size=10, keep_max_pool=False):
    # TODO: replace pretrained fc by new one?
    model = getattr(models, f"resnet{n}")(weights=getattr(models, f"ResNet{n}_Weights").DEFAULT)

    # adapt initial conv to input dimension
    conv1 = model.conv1
    kernel_size = 3
    conv1_new = nn.Conv2d(input_dim, conv1.out_channels, kernel_size, padding=(1, 1))
    weights_new = duplicate_weights(conv1.weight, input_dim, axis=1)[:, :, :kernel_size, :kernel_size]
    conv1_new.weight = nn.Parameter(weights_new)
    model.conv1 = conv1_new

    maxpool_original = model.maxpool

    def __change_strides(model, depth, max_depth, set_stride=(1, 1)):
        for name, layer in model.named_children():
            if isinstance(layer, (nn.Sequential, models.resnet.BasicBlock)):
                if depth < max_depth: __change_strides(layer, depth+1, max_depth)
            elif hasattr(layer, "stride"):
                setattr(layer, "stride", set_stride)

    # change strides because we can't afford shrinking
    __change_strides(model, 0, 10)

    if keep_max_pool: model.maxpool = maxpool_original
    
    # change fc to output dimension
    fc = model.fc
    fc_new = nn.Linear(fc.in_features, output_size, fc.bias is not None)
    weights_new = duplicate_weights(fc.weight, output_size, axis=0)
    fc.weight = nn.Parameter(weights_new)
    model.fc = fc_new

    return model


# definitions to load models through ArgumentParser and use in training script
def resnet18(n_classes, depth, **kwargs): return make_multispectral_resnet(18, input_dim=depth, output_size=n_classes)
def resnet50(n_classes, depth, **kwargs): return make_multispectral_resnet(50, input_dim=depth, output_size=n_classes)


if __name__ == "__main__":
    n_samples = 13
    input3 = torch.rand((n_samples, 3, 100, 100))
    input30 = torch.rand((n_samples, 30, 100, 100))

    resnet18 = make_multispectral_resnet(18)
    resnet50 = make_multispectral_resnet(50)

    import pdb
    pdb.set_trace()

