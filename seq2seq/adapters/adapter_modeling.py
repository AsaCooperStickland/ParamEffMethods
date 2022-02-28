"""Implements an Adapter, Low-rank adapters and Hyper-adapter Layers."""
import torch
import re
import torch.nn as nn
from .adapter_utils import Activations
from seq2seq.hypercomplex.layers import PHMLinear
from .low_rank_layer import LowRankLinear


class LowRankAdapter(nn.Module):
    """This is the low-rank adapter, in which each adapter is composed of two rank-one matrices.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = LowRankLinear(self.input_dim, self.down_sample_size,
                                          w_init=config.low_rank_w_init,
                                          rank=config.low_rank_rank)
        self.up_sampler = LowRankLinear(self.down_sample_size, self.input_dim,
                                        w_init=config.low_rank_w_init,
                                        rank=config.low_rank_rank)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output


class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = config.adapter_size
        self.activation = Activations(config.non_linearity.lower())
        if "glu" in config.non_linearity.lower():
            self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size * 2)
        else:
            self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size)
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output


class HyperComplexAdapter(nn.Module):
    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.activation = Activations(config.non_linearity.lower())
        self.down_sampler = PHMLinear(in_features=self.input_dim,
                                      out_features=self.down_sample_size,
                                      bias=True,
                                      c_init=config.phm_c_init,
                                      phm_dim=config.hypercomplex_division,
                                      learn_phm=config.learn_phm,
                                      w_init=config.hypercomplex_nonlinearity,
                                      shared_phm_rule=config.shared_phm_rule,
                                      factorized_phm=config.factorized_phm,
                                      shared_W_phm=config.shared_W_phm,
                                      factorized_phm_rule=config.factorized_phm_rule,
                                      phm_rank=config.phm_rank,
                                      phm_init_range=config.phm_init_range,
                                      kronecker_prod=config.kronecker_prod)
        self.up_sampler = PHMLinear(in_features=self.down_sample_size,
                                    out_features=self.input_dim,
                                    bias=True,
                                    c_init=config.phm_c_init,
                                    phm_dim=config.hypercomplex_division,
                                    learn_phm=config.learn_phm,
                                    w_init=config.hypercomplex_nonlinearity,
                                    shared_phm_rule=config.shared_phm_rule,
                                    factorized_phm=config.factorized_phm,
                                    shared_W_phm=config.shared_W_phm,
                                    factorized_phm_rule=config.factorized_phm_rule,
                                    phm_rank=config.phm_rank,
                                    phm_init_range=config.phm_init_range,
                                    kronecker_prod=config.kronecker_prod)
    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        return self.up_sampler(z)


class AddMultVec(nn.Module):
    def __init__(self, config, model_config, layer_name):
        super().__init__()
        self.input_dim = model_config.d_ff if layer_name in ["after_wi_ffn", "b4_wo_ffn"] else config.input_dim
        self.layer_list = config.layer_list.split(",")
        if "b4" in layer_name:
            after_name = re.sub("b4", "after", layer_name)
            self.use_add = config.use_add and after_name not in self.layer_list
        else:
            self.use_add = config.use_add
        self.use_mult = config.use_mult
        if self.use_add:
            self.add_hook = nn.Parameter(torch.zeros(self.input_dim))
        if self.use_mult:
            self.mult_hook = nn.Parameter(torch.ones(self.input_dim))

    def forward(self, x, condition=None):
        if self.use_mult:
            x = x * self.mult_hook
            if condition is not None:
                x = (1. + condition) * x
        if self.use_add:
            #print("add")
            if condition is not None:
                x  = x + condition * self.add_hook
            else:
                x = x + self.add_hook
            #print(self.add_hook.data)
        return x


class Distributor(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config, model_config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.layer_list = config.layer_list.split(",")
        self.use_add = config.use_add
        self.use_mult = config.use_mult
        self.hooks = nn.ModuleDict({layer: AddMultVec(config, model_config, layer)
                                   for layer in self.layer_list if (layer != "layer_norm"
                                   and not ("cross_attn" in layer and not model_config.is_decoder))})
        if config.condition_hooks:
            dim = len(self.hooks.keys())
            if "glu" in config.non_linearity.lower():
                dim = 2 * dim
            self.activation = Activations(config.non_linearity.lower())
            self.condition_layer = nn.Sequential(nn.Linear(self.input_dim, dim),
                                                self.activation)
            self.layer2id = {layer: i for layer, i in zip(self.hooks.keys(), range(dim))}
            self.hooks2 = nn.ModuleDict({layer: AddMultVec(config, model_config, layer)
                                       for layer in self.layer_list if (layer != "layer_norm"
                                       and not ("cross_attn" in layer and not model_config.is_decoder))})

    def forward(self, x, layer_name, condition=None):
        if layer_name in self.hooks.keys():
            if condition is not None:
                condition = condition[:, :, self.layer2id[layer_name]].unsqueeze(-1)
            x = self.hooks[layer_name](x, condition)
            if condition is not None:
                x = self.hooks2[layer_name](x, condition=None)
        return x
