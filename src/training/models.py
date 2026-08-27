"""Model factory + staged freeze/unfreeze control for transfer learning.

Both architectures share the same three-stage plan (see docs/training.md
for the full explanation in plain language):

  Stage 1 ("head only"): freeze the entire pretrained backbone, train only
  the new classification head.
  Stage 2 ("fine-tune last block"): unfreeze the backbone's last major
  block in addition to the head, at a lower learning rate.
  Stage 3 ("fine-tune more"): unfreeze one more block back. Only used if
  stage 2's validation results justify it - not run automatically.

ResNet18 and MobileNetV2 have very different internal structures, so "the
last block" means something architecture-specific in each case - documented
below rather than pretended to be identical. This is an approximation of
"as similar as possible," not a perfect match.
"""
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 2
MODEL_NAMES = ("resnet18", "mobilenet_v2")


def build_resnet18(pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_mobilenet_v2(pretrained=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model


_BUILDERS = {"resnet18": build_resnet18, "mobilenet_v2": build_mobilenet_v2}


def build_model(model_name, pretrained=True):
    if model_name not in _BUILDERS:
        raise ValueError(f"Unknown model_name {model_name!r}, expected one of {MODEL_NAMES}")
    return _BUILDERS[model_name](pretrained=pretrained)


def _stage_blocks(model, model_name):
    """Returns (head_module, {stage_number: [modules to unfreeze in addition
    to the head]}) for the given architecture.
    """
    if model_name == "resnet18":
        # ResNet18's four "layer" groups go from earliest (layer1, closest to
        # the input) to latest (layer4, closest to the head). Stage 2
        # unfreezes layer4; stage 3 additionally unfreezes layer3.
        return model.fc, {
            1: [],
            2: [model.layer4],
            3: [model.layer3, model.layer4],
        }
    if model_name == "mobilenet_v2":
        # model.features is a flat sequence of 19 inverted-residual/conv
        # blocks, earliest to latest. Stage 2 unfreezes the last 3 blocks
        # (roughly comparable in depth-fraction to ResNet18's layer4);
        # stage 3 unfreezes the last 7.
        feats = list(model.features)
        return model.classifier, {
            1: [],
            2: feats[-3:],
            3: feats[-7:],
        }
    raise ValueError(f"Unknown model_name {model_name!r}, expected one of {MODEL_NAMES}")


def set_stage(model, model_name, stage):
    """Freezes/unfreezes model parameters for the given stage (1, 2, or 3).
    Always trains the head; stage controls how much of the backbone is
    additionally unfrozen. Returns the model (mutated in place).
    """
    head, blocks_by_stage = _stage_blocks(model, model_name)
    if stage not in blocks_by_stage:
        raise ValueError(f"stage must be one of {sorted(blocks_by_stage)}, got {stage}")

    for p in model.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True
    for block in blocks_by_stage[stage]:
        for p in block.parameters():
            p.requires_grad = True

    return model


def trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_param_count(model):
    return sum(p.numel() for p in model.parameters())
