from collections import defaultdict
from typing import NamedTuple
from collections import OrderedDict

import copy
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from analysis.metric import get_l2, get_flatness
from models.resnet import BasicBlock


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def mlp_permutation_spec(num_hidden_layers: int, bias_flg: bool) -> PermutationSpec:
    """We assume that one permutation cannot appear in two axes of the same weight array."""
    assert num_hidden_layers >= 1
    if bias_flg:
        return permutation_spec_from_axes_to_perm({
            "layer0.weight": ("P_0", None),
            **{f"layer{i}.weight": (f"P_{i}", f"P_{i - 1}")
               for i in range(1, num_hidden_layers)},
            **{f"layer{i}.bias": (f"P_{i}",)
               for i in range(num_hidden_layers)},
            f"layer{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers - 1}"),
            f"layer{num_hidden_layers}.bias": (None,),
        })
    else:
        return permutation_spec_from_axes_to_perm({
            "layer0.weight": ("P_0", None),
            **{f"layer{i}.weight": (f"P_{i}", f"P_{i - 1}")
               for i in range(1, num_hidden_layers)},
            f"layer{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers - 1}"),
        })


def cnn_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,)}
    dense = lambda name, p_in, p_out, bias=True: {f"{name}.weight": (p_out, p_in),
                                                  f"{name}.bias": (p_out,)} if bias else {
        f"{name}.weight": (p_out, p_in)}

    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P_bg0"),
        **conv("conv2", "P_bg0", "P_bg1"),
        **dense("fc1", "P_bg1", "P_bg2"),
        **dense("fc2", "P_bg2", "P_bg3"),
        **dense("fc3", "P_bg3", None),
    })


def resnet20_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_out, p_in: {f"{name}.weight": (p_out, p_in, None, None, None)}
    norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, ), f"{name}.running_mean": (p, ), f"{name}.running_var": (p, )}
    dense = lambda name, p_out, p_in: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}
    # This is for easy blocks that use a residual connection, without any change in the number of channels.
    shortcutblock_ns = lambda name, p_out, p_in: {
    **conv(f"{name}.conv1", f"P_{name}_inner", p_in),
    **norm(f"{name}.bn1", f"P_{name}_inner"),
    **conv(f"{name}.conv2", p_out, f"P_{name}_inner"),
    **norm(f"{name}.bn2", p_out),
    }
    # This is for blocks that use a residual connection, but change the number of channels via a Conv.
    shortcutblock = lambda name, p_out, p_in: {
    **conv(f"{name}.conv1", f"P_{name}_inner", p_in),
    **norm(f"{name}.bn1", f"P_{name}_inner"),
    **conv(f"{name}.conv2", p_out, f"P_{name}_inner"),
    **norm(f"{name}.bn2", p_out),
    **conv(f"{name}.shortcut.0", p_out, p_in),
    **norm(f"{name}.shortcut.1", p_out),
    }
    #"P_bg0"
    return permutation_spec_from_axes_to_perm({
      **conv("conv1", "P_bg0", None),
      **norm("bn1", "P_bg0"),
      **shortcutblock_ns("layer1.0", "P_bg0", "P_bg0"),
      **shortcutblock_ns("layer1.1", "P_bg0", "P_bg0"),
      **shortcutblock_ns("layer1.2", "P_bg0", "P_bg0"),
      **shortcutblock("layer2.0", "P_bg2.0", "P_bg0"),
      **shortcutblock_ns("layer2.1", "P_bg2.0", "P_bg2.0"),
      **shortcutblock_ns("layer2.2", "P_bg2.0", "P_bg2.0"),
      **shortcutblock("layer3.0", "P_bg3.0", "P_bg2.0"),
      **shortcutblock_ns("layer3.1", "P_bg3.0", "P_bg3.0"),
      **shortcutblock_ns("layer3.2", "P_bg3.0", "P_bg3.0"),
      **dense("linear", None, "P_bg3.0"),
})


def resnet20_i_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_out, p_in: {f"{name}.weight": (p_out, p_in, None, None, None)}
    norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, ), f"{name}.running_mean": (p, ), f"{name}.running_var": (p, )}
    dense = lambda name, p_out, p_in: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}
    identity = lambda name, p_out, p_in: {f"{name}.identity": (p_out, p_in)}
    # This is for easy blocks that use a residual connection, without any change in the number of channels.
    shortcutblock_ns = lambda name, p_out, p_in: {
    **conv(f"{name}.conv1", f"P_{name}_inner1", p_in),
    **norm(f"{name}.bn1", f"P_{name}_inner1"),
    **conv(f"{name}.conv2", p_out, f"P_{name}_inner1"),
    **norm(f"{name}.bn2", p_out),
    **identity(f"{name}.shortcut.1", p_out, p_in)
    }
    # This is for blocks that use a residual connection, but change the number of channels via a Conv.
    shortcutblock = lambda name, p_out, p_in: {
    **conv(f"{name}.conv1", f"P_{name}_inner1", p_in),
    **norm(f"{name}.bn1", f"P_{name}_inner1"),
    **conv(f"{name}.conv2", p_out, f"P_{name}_inner1"),
    **norm(f"{name}.bn2", p_out),
    **conv(f"{name}.shortcut.0.0", f"P_{name}_inner2", p_in),
    **norm(f"{name}.shortcut.0.1", f"P_{name}_inner2"),
    **identity(f"{name}.shortcut.1", p_out, f"P_{name}_inner2")
    }
    #"P_bg0"
    return permutation_spec_from_axes_to_perm({
      **conv("conv1", "P_bg0", None),
      **norm("bn1", "P_bg0"),
      **shortcutblock_ns("layer1.0", "P_bg1.0", "P_bg0"),
      **shortcutblock_ns("layer1.1", "P_bg1.1", "P_bg1.0"),
      **shortcutblock_ns("layer1.2", "P_bg1.2", "P_bg1.1"),
      **shortcutblock("layer2.0", "P_bg2.0", "P_bg1.2"),
      **shortcutblock_ns("layer2.1", "P_bg2.1", "P_bg2.0"),
      **shortcutblock_ns("layer2.2", "P_bg2.2", "P_bg2.1"),
      **shortcutblock("layer3.0", "P_bg3.0", "P_bg2.2"),
      **shortcutblock_ns("layer3.1", "P_bg3.1", "P_bg3.0"),
      **shortcutblock_ns("layer3.2", "P_bg3.2", "P_bg3.1"),
      **dense("linear", None, "P_bg3.2"),
})


# should be easy to generalize it to any depth
def resnet50_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None,)}
    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
    dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}

    # This is for easy blocks that use a residual connection, without any change in the number of channels.
    easyblock = lambda name, p: {
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(f"{name}.bn1", p),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
    }

    # This is for blocks that use a residual connection, but change the number of channels via a Conv.
    shortcutblock = lambda name, p_in, p_out: {
        **norm(f"{name}.bn1", p_in),
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **conv(f"{name}.shortcut.0", p_in, p_out),
        **norm(f"{name}.shortcut.1", p_out),
    }

    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, "P_bg0"),
        **norm("bn1", "P_bg0"),

        **shortcutblock("layer1.0", "P_bg0", "P_bg1"),
        **easyblock("layer1.1", "P_bg1", ),
        **easyblock("layer1.2", "P_bg1"),
        **easyblock("layer1.3", "P_bg1"),
        **easyblock("layer1.4", "P_bg1"),
        **easyblock("layer1.5", "P_bg1"),
        **easyblock("layer1.6", "P_bg1"),
        **easyblock("layer1.7", "P_bg1"),

        # **easyblock("layer1.3", "P_bg1"),

        **shortcutblock("layer2.0", "P_bg1", "P_bg2"),
        **easyblock("layer2.1", "P_bg2", ),
        **easyblock("layer2.2", "P_bg2"),
        **easyblock("layer2.3", "P_bg2"),
        **easyblock("layer2.4", "P_bg2"),
        **easyblock("layer2.5", "P_bg2"),
        **easyblock("layer2.6", "P_bg2"),
        **easyblock("layer2.7", "P_bg2"),

        **shortcutblock("layer3.0", "P_bg2", "P_bg3"),
        **easyblock("layer3.1", "P_bg3", ),
        **easyblock("layer3.2", "P_bg3"),
        **easyblock("layer3.3", "P_bg3"),
        **easyblock("layer3.4", "P_bg3"),
        **easyblock("layer3.5", "P_bg3"),
        **easyblock("layer3.6", "P_bg3"),
        **easyblock("layer3.7", "P_bg3"),

        **dense("linear", "P_bg3", None),

    })


def vgg11_permutation_spec() -> PermutationSpec:
    return permutation_spec_from_axes_to_perm({
                "features.0.weight": ("P_Conv_0", None, None, None),
                "features.0.bias": ("P_Conv_0", None),
                "features.1.weight": ("P_Conv_0", None),
                "features.1.bias": ("P_Conv_0", None),
                "features.1.running_mean": ("P_Conv_0", None),
                "features.1.running_var": ("P_Conv_0", None),
                "features.1.num_batches_tracked": ("P_Conv_0", None),

                "features.4.weight": ("P_Conv_1", "P_Conv_0", None, None),
                "features.4.bias": ("P_Conv_1", None),
                "features.5.weight": ("P_Conv_1", None),
                "features.5.bias": ("P_Conv_1", None),
                "features.5.running_mean": ("P_Conv_1", None),
                "features.5.running_var": ("P_Conv_1", None),
                "features.5.num_batches_tracked": ("P_Conv_1", None),

                "features.8.weight": ("P_Conv_2", "P_Conv_1", None, None),
                "features.8.bias": ("P_Conv_2", None),
                "features.9.weight": ("P_Conv_2", None),
                "features.9.bias": ("P_Conv_2", None),
                "features.9.running_mean": ("P_Conv_2", None),
                "features.9.running_var": ("P_Conv_2", None),
                "features.9.num_batches_tracked": ("P_Conv_2", None),

                "features.11.weight": ("P_Conv_3", "P_Conv_2", None, None),
                "features.11.bias": ("P_Conv_3", None),
                "features.12.weight": ("P_Conv_3", None),
                "features.12.bias": ("P_Conv_3", None),
                "features.12.running_mean": ("P_Conv_3", None),
                "features.12.running_var": ("P_Conv_3", None),
                "features.12.num_batches_tracked": ("P_Conv_3", None),

                "features.15.weight": ("P_Conv_4", "P_Conv_3", None, None),
                "features.15.bias": ("P_Conv_4", None),
                "features.16.weight": ("P_Conv_4", None),
                "features.16.bias": ("P_Conv_4", None),
                "features.16.running_mean": ("P_Conv_4", None),
                "features.16.running_var": ("P_Conv_4", None),
                "features.16.num_batches_tracked": ("P_Conv_4", None),

                "features.18.weight": ("P_Conv_5", "P_Conv_4", None, None),
                "features.18.bias": ("P_Conv_5", None),
                "features.19.weight": ("P_Conv_5", None),
                "features.19.bias": ("P_Conv_5", None),
                "features.19.running_mean": ("P_Conv_5", None),
                "features.19.running_var": ("P_Conv_5", None),
                "features.19.num_batches_tracked": ("P_Conv_5", None),

                "features.22.weight": ("P_Conv_6", "P_Conv_5", None, None),
                "features.22.bias": ("P_Conv_6", None),
                "features.23.weight": ("P_Conv_6", None),
                "features.23.bias": ("P_Conv_6", None),
                "features.23.running_mean": ("P_Conv_6", None),
                "features.23.running_var": ("P_Conv_6", None),
                "features.23.num_batches_tracked": ("P_Conv_6", None),

                "features.25.weight": ("P_Conv_7", "P_Conv_6", None, None),
                "features.25.bias": ("P_Conv_7", None),
                "features.26.weight": ("P_Conv_7", None),
                "features.26.bias": ("P_Conv_7", None),
                "features.26.running_mean": ("P_Conv_7", None),
                "features.26.running_var": ("P_Conv_7", None),
                "features.26.num_batches_tracked": ("P_Conv_7", None),

                "classifier.weight": (None, "P_Conv_7"),
                "classifier.bias": (None, None),
    })


def vgg16_permutation_spec() -> PermutationSpec:
    return permutation_spec_from_axes_to_perm({
                "features.0.weight": ("P_Conv_0", None, None, None),
                "features.0.bias": ("P_Conv_0", None),
                "features.1.weight": ("P_Conv_0", None),
                "features.1.bias": ("P_Conv_0", None),

                "features.3.weight": ("P_Conv_1", "P_Conv_0", None, None),
                "features.3.bias": ("P_Conv_1", None),
                "features.4.weight": ("P_Conv_1", None),
                "features.4.bias": ("P_Conv_1", None),

                "features.7.weight": ("P_Conv_2", "P_Conv_1", None, None),
                "features.7.bias": ("P_Conv_2", None),
                "features.8.weight": ("P_Conv_2", None),
                "features.8.bias": ("P_Conv_2", None),

                "features.10.weight": ("P_Conv_3", "P_Conv_2", None, None),
                "features.10.bias": ("P_Conv_3", None),
                "features.11.weight": ("P_Conv_3", None),
                "features.11.bias": ("P_Conv_3", None),

                "features.14.weight": ("P_Conv_4", "P_Conv_3", None, None),
                "features.14.bias": ("P_Conv_4", None),
                "features.15.weight": ("P_Conv_4", None),
                "features.15.bias": ("P_Conv_4", None),

                "features.17.weight": ("P_Conv_5", "P_Conv_4", None, None),
                "features.17.bias": ("P_Conv_5", None),
                "features.18.weight": ("P_Conv_5", None),
                "features.18.bias": ("P_Conv_5", None),

                "features.20.weight": ("P_Conv_6", "P_Conv_5", None, None),
                "features.20.bias": ("P_Conv_6", None),
                "features.21.weight": ("P_Conv_6", None),
                "features.21.bias": ("P_Conv_6", None),

                "features.24.weight": ("P_Conv_7", "P_Conv_6", None, None),
                "features.24.bias": ("P_Conv_7", None),
                "features.25.weight": ("P_Conv_7", None),
                "features.25.bias": ("P_Conv_7", None),

                "features.27.weight": ("P_Conv_8", "P_Conv_7", None, None),
                "features.27.bias": ("P_Conv_8", None),
                "features.28.weight": ("P_Conv_8", None),
                "features.28.bias": ("P_Conv_8", None),

                "features.30.weight": ("P_Conv_9", "P_Conv_8", None, None),
                "features.30.bias": ("P_Conv_9", None),
                "features.31.weight": ("P_Conv_9", None),
                "features.31.bias": ("P_Conv_9", None),

                "features.34.weight": ("P_Conv_10", "P_Conv_9", None, None),
                "features.34.bias": ("P_Conv_10", None),
                "features.35.weight": ("P_Conv_10", None),
                "features.35.bias": ("P_Conv_10", None),

                "features.37.weight": ("P_Conv_11", "P_Conv_10", None, None),
                "features.37.bias": ("P_Conv_11", None),
                "features.38.weight": ("P_Conv_11", None),
                "features.38.bias": ("P_Conv_11", None),

                "features.40.weight": ("P_Conv_12", "P_Conv_11", None, None),
                "features.40.bias": ("P_Conv_12", None),
                "features.41.weight": ("P_Conv_12", None),
                "features.41.bias": ("P_Conv_12", None),

                "classifier0.weight": ("P_Dence_0", "P_Conv_12"),
                "classifier0.bias": ("P_Dence_0", None),

                "classifier1.weight": ("P_Dence_1", "P_Dence_0"),
                "classifier1.bias": ("P_Dence_1", None),

                "classifier.weight": (None, "P_Dence_1"),
                "classifier.bias": (None, None),
    })


def vgg1_16_permutation_spec() -> PermutationSpec:
    return permutation_spec_from_axes_to_perm({
                "features.0.weight": ("P_Conv_0", None, None, None),
                "features.0.bias": ("P_Conv_0", None),
                "features.1.weight": ("P_Conv_0", None),
                "features.1.bias": ("P_Conv_0", None),

                "features.3.weight": ("P_Conv_1", "P_Conv_0", None, None),
                "features.3.bias": ("P_Conv_1", None),
                "features.4.weight": ("P_Conv_1", None),
                "features.4.bias": ("P_Conv_1", None),

                "features.7.weight": ("P_Conv_2", "P_Conv_1", None, None),
                "features.7.bias": ("P_Conv_2", None),
                "features.8.weight": ("P_Conv_2", None),
                "features.8.bias": ("P_Conv_2", None),

                "features.10.weight": ("P_Conv_3", "P_Conv_2", None, None),
                "features.10.bias": ("P_Conv_3", None),
                "features.11.weight": ("P_Conv_3", None),
                "features.11.bias": ("P_Conv_3", None),

                "features.14.weight": ("P_Conv_4", "P_Conv_3", None, None),
                "features.14.bias": ("P_Conv_4", None),
                "features.15.weight": ("P_Conv_4", None),
                "features.15.bias": ("P_Conv_4", None),

                "features.17.weight": ("P_Conv_5", "P_Conv_4", None, None),
                "features.17.bias": ("P_Conv_5", None),
                "features.18.weight": ("P_Conv_5", None),
                "features.18.bias": ("P_Conv_5", None),

                "features.20.weight": ("P_Conv_6", "P_Conv_5", None, None),
                "features.20.bias": ("P_Conv_6", None),
                "features.21.weight": ("P_Conv_6", None),
                "features.21.bias": ("P_Conv_6", None),

                "features.24.weight": ("P_Conv_7", "P_Conv_6", None, None),
                "features.24.bias": ("P_Conv_7", None),
                "features.25.weight": ("P_Conv_7", None),
                "features.25.bias": ("P_Conv_7", None),

                "features.27.weight": ("P_Conv_8", "P_Conv_7", None, None),
                "features.27.bias": ("P_Conv_8", None),
                "features.28.weight": ("P_Conv_8", None),
                "features.28.bias": ("P_Conv_8", None),

                "features.30.weight": ("P_Conv_9", "P_Conv_8", None, None),
                "features.30.bias": ("P_Conv_9", None),
                "features.31.weight": ("P_Conv_9", None),
                "features.31.bias": ("P_Conv_9", None),

                "features.34.weight": ("P_Conv_10", "P_Conv_9", None, None),
                "features.34.bias": ("P_Conv_10", None),
                "features.35.weight": ("P_Conv_10", None),
                "features.35.bias": ("P_Conv_10", None),

                "features.37.weight": ("P_Conv_11", "P_Conv_10", None, None),
                "features.37.bias": ("P_Conv_11", None),
                "features.38.weight": ("P_Conv_11", None),
                "features.38.bias": ("P_Conv_11", None),

                "features.40.weight": ("P_Conv_12", "P_Conv_11", None, None),
                "features.40.bias": ("P_Conv_12", None),
                "features.41.weight": ("P_Conv_12", None),
                "features.41.bias": ("P_Conv_12", None),

                "classifier.weight": (None, "P_Conv_12"),
                "classifier.bias": (None, None),

    })


class IdentityWithParam(nn.Module):
    def __init__(self, in_channels):
        super(IdentityWithParam, self).__init__()
        self.identity = nn.Parameter(torch.eye(in_channels), requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.size()
        x_reshaped = x.view(b, c, h * w)

        # Repeat self.identity 'b' times to match batch size
        identity_expanded = self.identity.repeat(b, 1, 1)

        out = torch.bmm(identity_expanded, x_reshaped)
        out = out.view(b, c, h, w)
        return out


def add_identity_shortcut_recursive(module):
    if isinstance(module, BasicBlock):
        if not isinstance(module.shortcut, IdentityWithParam):
            original_shortcut = module.shortcut
            module.shortcut = nn.Sequential(
                original_shortcut,
                IdentityWithParam(module.conv2.out_channels)
            )
    else:
        for child in module.children():
            add_identity_shortcut_recursive(child)


def remove_identity_shortcut_recursive(module):
    if isinstance(module, BasicBlock):
        if isinstance(module.shortcut, nn.Sequential) and \
                any(isinstance(layer, IdentityWithParam) for layer in module.shortcut):
            layers = [layer for layer in module.shortcut if not isinstance(layer, IdentityWithParam)]
            module.shortcut = nn.Sequential(*layers)
    else:
        for child in module.children():
            remove_identity_shortcut_recursive(child)


def add_identity_to_net(original_net):
    net = copy.deepcopy(original_net)
    add_identity_shortcut_recursive(net)
    return net


def remove_identity_from_net(original_net):
    net = copy.deepcopy(original_net)
    remove_identity_shortcut_recursive(net)
    return net


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]

    # if k == 'classifier.weight':  # to reshape because of input shape is 3x 96 x 96
    #     w = w.reshape(126, 512 * 4, 3, 3)
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            w = torch.index_select(w, axis, perm[p].int())
    # if k == 'classifier.weight':
    #     w = w.reshape(126, -1)
    return w


def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply a `perm` to `params`."""
    ret = {}
    for k in params.keys():
        if params[k].dim() != 0:  # avoid num_batches_tracked
            ret[k] = get_permuted_param(ps, perm, k, params)
        else:
            ret[k] = params[k]
    return ret


def weight_matching(ps: PermutationSpec, params_a, params_b, max_iter=300, init_perm=None, print_flg=True):
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    device = list(params_a.values())[0].device
    perm = {p: torch.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
    perm = {key: perm[key].cpu() for key in perm}  # to cpu
    params_a = {key: params_a[key].cpu() for key in params_a}  # to cpu
    params_b = {key: params_b[key].cpu() for key in params_b}  # to cpu
    perm_names = list(perm.keys())
    metrics = {'step': [], 'l2_dist': []}
    step = 0
    for iteration in range(max_iter):
        progress = False
        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = torch.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:  # layer loop
                if ('running_mean' not in wk) and ('running_var' not in wk) and ('num_batches_tracked' not in wk):
                    w_a = params_a[wk]  # target
                    w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                    w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                    w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))

                    A += w_a @ w_b.T  # A is cost matrix to assignment,
            # https://github.com/scipy/scipy/blob/v0.18.1/scipy/optimize/_hungarian.py#L13-L107
            ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)
            # ri, ci = linear_sum_assignment(A, maximize=True)
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()
            oldL = torch.einsum('ij,ij->i', A, torch.eye(n)[perm[p].long()]).sum()
            newL = torch.einsum('ij,ij->i', A, torch.eye(n)[ci, :]).sum()
            if print_flg:
                print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = torch.Tensor(ci)
            p_params_b = apply_permutation(ps, perm, params_b)
            l2_dist = get_l2(params_a, p_params_b)
            metrics['step'].append(step)
            metrics['l2_dist'].append(l2_dist)
            step += 1
        if not progress:
            break
    perm = {key: perm[key].to(device) for key in perm}  # to device
    params_a = {key: params_a[key].to(device) for key in params_a}  # to device
    params_b = {key: params_b[key].to(device) for key in params_b}  # to device
    return perm, metrics


def flat_weight_matching(ps: PermutationSpec, params_a, params_b, grad_a, grad_b, lam=1.0,  max_iter=300, init_perm=None, print_flg=True):
    """Find a permutation of `params_b` """
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    device = list(params_a.values())[0].device
    perm = {p: torch.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
    perm = {key: perm[key].cpu() for key in perm}  # to cpu
    params_a = {key: params_a[key].cpu() for key in params_a}  # to cpu
    params_b = {key: params_b[key].cpu() for key in params_b}  # to cpu
    grad_a = {key: grad_a[key].cpu() for key in grad_a}  # to cpu
    grad_b = {key: grad_b[key].cpu() for key in grad_b}  # to cpu
    perm_names = list(perm.keys())
    metrics = {'step': [], 'l2_dist': [], 'flatness': []}
    step = 0

    l2_dist = get_l2(params_a, params_b)
    flatness = get_flatness(params_a, params_b, grad_b)
    metrics['flatness'].append(flatness)
    metrics['step'].append(step)
    metrics['l2_dist'].append(l2_dist)

    for iteration in range(max_iter):
        progress = False

        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = torch.zeros((n, n), device=device)
            for wk, axis in ps.perm_to_axes[p]:  # layer loop
                if ('running_mean' not in wk) and ('running_var' not in wk) and ('num_batches_tracked' not in wk):
                    # pram
                    w_a = params_a[wk]  # target
                    #
                    w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                    w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                    w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))
                    A += lam * w_a @ w_b.T  # A is cost matrix to assignment

                    # grad
                    g_b = get_permuted_param(ps, perm, wk, grad_b, except_axis=axis)
                    g_b = torch.moveaxis(g_b, axis, 0).reshape((n, -1))
                    A -= (1 - lam) * torch.einsum('ij,kj->ik', w_a, g_b)

            # https://github.com/scipy/scipy/blob/v0.18.1/scipy/optimize/_hungarian.py#L13-L107
            # ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)
            ri, ci = linear_sum_assignment(A, maximize=True)
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()
            oldL = torch.einsum('ij,ij->i', A, torch.eye(n)[perm[p].long()]).sum()
            newL = torch.einsum('ij,ij->i', A, torch.eye(n)[ci, :]).sum()
            if print_flg:
                print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = torch.Tensor(ci)
            p_params_b = apply_permutation(ps, perm, params_b)
            p_grad_b = apply_permutation(ps, perm, grad_b)
            l2_dist = get_l2(params_a, p_params_b)
            flatness = get_flatness(params_a, p_params_b, p_grad_b)
            metrics['flatness'].append(flatness)
            metrics['step'].append(step)
            metrics['l2_dist'].append(l2_dist)
            step += 1
        if not progress:
            break
    perm = {key: perm[key].to(device) for key in perm}  # to device
    params_a = {key: params_a[key].to(device) for key in params_a}  # to device
    params_b = {key: params_b[key].to(device) for key in params_b}  # to device
    grad_a = {key: grad_a[key].to(device) for key in grad_a}  # to cpu
    grad_b = {key: grad_b[key].to(device) for key in grad_b}  # to cpu
    return perm, metrics


def flat_weight_matching_v2(ps: PermutationSpec, params_a, params_b, grad_a, grad_b, lam=1.0,  max_iter=300, init_perm=None, print_flg=True):
    """Find a permutation of `params_b` """
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    device = list(params_a.values())[0].device
    perm = {p: torch.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
    perm = {key: perm[key].cpu() for key in perm}  # to cpu
    params_a = {key: params_a[key].cpu() for key in params_a}  # to cpu
    params_b = {key: params_b[key].cpu() for key in params_b}  # to cpu
    grad_a = {key: grad_a[key].cpu() for key in grad_a}  # to cpu
    grad_b = {key: grad_b[key].cpu() for key in grad_b}  # to cpu
    perm_names = list(perm.keys())
    metrics = {'step': [], 'l2_dist': [], 'flatness': []}
    step = 0

    l2_dist = get_l2(params_a, params_b)
    flatness = get_flatness(params_a, params_b, grad_b)
    metrics['flatness'].append(flatness)
    metrics['step'].append(step)
    metrics['l2_dist'].append(l2_dist)

    for iteration in range(max_iter):
        progress = False

        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = torch.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                if ('running_mean' not in wk) and ('running_var' not in wk) and ('num_batches_tracked' not in wk) and ('identity' not in wk):
                    w_a = params_a[wk]  # w_a_
                    w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                    w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                    w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))
                    A += 2 * lam * w_a @ w_b.T  # A is cost matrix to assignment
                    # grad
                    g_a = grad_a[wk]
                    g_a = torch.moveaxis(g_a, axis, 0).reshape((n, -1))
                    A -= (1 - lam) * 0.5 * g_a @ w_b.T

            # https://github.com/scipy/scipy/blob/v0.18.1/scipy/optimize/_hungarian.py#L13-L107
            ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()
            oldL = torch.einsum('ij,ij->i', A, torch.eye(n)[perm[p].long()]).sum()
            newL = torch.einsum('ij,ij->i', A, torch.eye(n)[ci, :]).sum()
            if print_flg:
                print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = torch.Tensor(ci)
            p_params_b = apply_permutation(ps, perm, params_b)
            l2_dist = get_l2(params_a, p_params_b)
            p_grad_b = apply_permutation(ps, perm, grad_b)
            flatness = get_flatness(params_a, p_params_b, p_grad_b)
            metrics['flatness'].append(flatness)
            metrics['step'].append(step)
            metrics['l2_dist'].append(l2_dist)
            step += 1
        if not progress:
            break
    perm = {key: perm[key].to(device) for key in perm}  # to device
    params_a = {key: params_a[key].to(device) for key in params_a}  # to device
    params_b = {key: params_b[key].to(device) for key in params_b}  # to device
    grad_a = {key: grad_a[key].to(device) for key in grad_a}  # to cpu
    grad_b = {key: grad_b[key].to(device) for key in grad_b}  # to cpu
    return perm, metrics


class StoreBN(object):
    def __init__(self):
        self.full_key = None
        self.state_dict = None

    def remove_bn(self, params):
        self.full_key = list(params.keys())
        self.state_dict = OrderedDict()  # init
        for key in self.full_key:
            if ('running_var' in key) or ('running_mean' in key):
                self.state_dict[key] = params[key]
                del params[key]
        return params

    def repair_bn(self, params):
        for key in self.full_key:
            if ('running_var' in key) or ('running_mean' in key):
                params[key] = self.state_dict[key]
            else:
                params[key] = params[key]
        return params


def test_weight_matching_mlp():
    """If we just have a single hidden layer then it should converge after just one step."""
    ps = mlp_permutation_spec(num_hidden_layers=3, bias_flg=True)
    print(ps.axes_to_perm)
    rng = torch.Generator()
    rng.manual_seed(13)
    num_hidden = 10
    shapes = {
        "layer0.weight": (2, num_hidden),
        "layer0.bias": (num_hidden,),
        "layer1.weight": (num_hidden, 3),
        "layer1.bias": (3,)
    }

    params_a = {k: torch.randn(shape, generator=rng) for k, shape in shapes.items()}
    params_b = {k: torch.randn(shape, generator=rng) for k, shape in shapes.items()}
    perm = weight_matching(rng, ps, params_a, params_b)
    print(perm)


def test_weight_matching_vgg1():
    rng = torch.Generator()
    rng.manual_seed(13)
    from models.vgg import VGG
    model = VGG('VGG16', 10)
    ps = vgg1_16_permutation_spec()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)


    perm_sizes = {p: model.state_dict()[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    perm = {p: torch.randperm(n) for p, n in perm_sizes.items()}
    updated_params = apply_permutation(ps, perm, model.state_dict())
    model.load_state_dict(updated_params)
    perm_y = model(x)
    print((y - perm_y) / perm_y.min())


def test_weight_matching_vgg():
    rng = torch.Generator()
    rng.manual_seed(13)
    from models.vgg import VGG
    model = VGG('VGG11', 10)
    # model = VGG('VGG11', 126, w=4, input_shape=3*96*96)
    ps = vgg11_permutation_spec()
    x = torch.randn(2, 3, 32, 32) * 100
    # x = torch.randn(2, 3, 96, 96) * 100
    y = model(x)

    perm_sizes = {p: model.state_dict()[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    perm = {p: torch.randperm(n) for p, n in perm_sizes.items()}
    updated_params = apply_permutation(ps, perm, model.state_dict())
    model.load_state_dict(updated_params)
    perm_y = model(x)
    print((y - perm_y) / perm_y.min())


def test_weight_matching_resnet():
    rng = torch.Generator()
    rng.manual_seed(21)
    from models.resnet import resnet20
    model = resnet20(w=4, num_classes=10)
    ps = resnet20_permutation_spec()
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)

    perm_sizes = {p: model.state_dict()[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    perm = {p: torch.randperm(n) for p, n in perm_sizes.items()}
    updated_params = apply_permutation(ps, perm, model.state_dict())
    model.load_state_dict(updated_params)
    model.eval()
    perm_y = model(x)
    print((y - perm_y) / perm_y.mean())


def test_weight_perm_mlp():
    from copy import deepcopy
    rng = torch.Generator()
    rng.manual_seed(21)
    from models.mlp import MLP
    model = MLP()
    model.eval()
    x = torch.randn(2, 28, 28)
    y = model(x)
    ps = mlp_permutation_spec(num_hidden_layers=3, bias_flg=True)
    perm_sizes = {p: model.state_dict()[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    perm = {p: torch.randperm(n) for p, n in perm_sizes.items()}
    updated_params = apply_permutation(ps, perm, model.state_dict())
    origin_params = deepcopy(model.state_dict())
    model.load_state_dict(updated_params)
    model.eval()
    perm_y = model(x)
    print((y - perm_y) / perm_y.mean())

    # for key in origin_params:
    #     print(origin_params[key] - updated_params[key])

if __name__ == "__main__":
    # test_weight_perm_mlp()
    # test_weight_matching_mlp()
    # test_weight_matching_vgg()
    test_weight_matching_resnet()
