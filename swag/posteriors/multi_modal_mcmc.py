from os import stat
from random import sample
from torch.cuda import init
import torch.nn as nn
import torch


class cSGMCMC(nn.Module):
    def __init__(
        self,
        state_dicts,
        base_model,
        num_modes,
        samples_per_mode,
    ) -> None:
        super().__init__()
        self.state_dicts = state_dicts
        self.base_model = base_model
        self.num_modes = num_modes
        self.samples_per_mode = samples_per_mode

    def _load_sample(self, mode, sample_num):
        sd = self.state_dicts[mode][sample_num]
        checkpoint = torch.load(sd)
        self.base_model.load_state_dict(checkpoint["state_dict"])

    def forward(self, x):
        self._load_sample(0, 0)
        counter = 0
        tmp = self.base_model(x)
        out = torch.zeros_like(tmp).to(tmp.device)
        for i in range(self.num_modes):
            for j in range(self.samples_per_mode):
                self._load_sample(i, j)
                out += self.base_model(x)
                counter += 1
        return out / counter


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
