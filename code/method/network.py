import torch
from torch import nn
from torch.nn import Parameter


class WassersteinDiscriminatorSN(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(WassersteinDiscriminatorSN, self).__init__()
        self.ad_layer1 = SpectralNorm(nn.Linear(in_feature, hidden_size))
        self.ad_layer2 = SpectralNorm(nn.Linear(hidden_size, hidden_size))
        self.ad_layer3 = SpectralNorm(nn.Linear(hidden_size, 1))
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.apply(self.init_weight_)

    def init_weight_(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)
        elif classname.find('SpectralNorm') != -1:
            nn.init.kaiming_uniform_(m.module.weight_bar)
            nn.init.zeros_(m.module.bias)

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        y = self.ad_layer3(x)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)
