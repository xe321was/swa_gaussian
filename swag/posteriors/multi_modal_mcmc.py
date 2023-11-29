from os import stat
from random import sample
from torch.cuda import init
import torch.nn as nn
import torch


class cSGMCMC(nn.Module):
    def __init__(
        self,
        mode_models,
    ) -> None:
        super().__init__()
        self.mode_models = mode_models
        self.num_modes = len(mode_models)
        self.samples_per_mode = len(mode_models[0])

    def forward(self, x):
        output_list = []
        for i in range(self.num_modes):
            for j in range(self.samples_per_mode):
                model = self.mode_models[i][j].cuda()
                model.cuda()
                model.eval()
                output_list.append(model(x))
        return sum(output_list) / len(output_list)


# TODO: how to make this so that adversarial attacker can use information from all the modes when attacking?
# Idea: store on the cpu, move to cuda only during eval!
# initialize the correct number of SWAG models, and pass into here
class cSWAG(nn.Module):
    def __init__(self, mode_models, cov, scale, bn_update_fn) -> None:
        super().__init__()
        self.mode_models = mode_models
        self.num_modes = len(mode_models)
        self.cov = cov
        self.scale = scale
        self.bn_update_fn = bn_update_fn

    def update_models(self):
        for model in self.mode_models:
            model.train()
            model.cuda()
            model.sample(scale=self.scale, cov=self.cov)
            self.bn_update_fn(model)
            model.cpu()

    def forward(self, x):
        output_list = []
        for model in self.mode_models:
            model.cuda()
            model.eval()
            output_list.append(model(x))
            model.cpu()
        return sum(output_list) / len(output_list)
