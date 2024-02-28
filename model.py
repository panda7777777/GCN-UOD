import torch.nn as nn


class Estimator(nn.Module):
    def __init__(self, ngpu, latent_size):
        super(Estimator, self).__init__()
        self.ngpu = ngpu
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size // 2),
            nn.ReLU(),
            nn.Linear(self.latent_size // 2, self.latent_size // 4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size // 4, self.latent_size // 2),
            nn.ReLU(),
            nn.Linear(self.latent_size // 2, self.latent_size),
            nn.Sigmoid()
        )

        self.confidence = nn.Sequential(
            nn.Linear(self.latent_size // 4, 1),
            nn.Sigmoid()
        )

        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'activation_fn'):
                if m.activation_fn == 'tanh':
                    nn.init.xavier_normal_(m.weight.data)
                elif m.activation_fn == 'relu':
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
                elif m.activation_fn == 'sigmoid':
                    nn.init.xavier_normal_(m.weight.data)
            else:
                nn.init.normal_(m.weight.data, mean=0, std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    def forward(self, input):
        features = self.encoder(input)
        rec = self.decoder(features)
        conf = self.confidence(features)

        return rec, conf


class Generator(nn.Module):
    def __init__(self, ngpu, latent_size):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        # Main sequence of layers
        self.main = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size, bias=True),
            nn.ReLU(),
            nn.Linear(2 * latent_size, latent_size, bias=True),
            nn.Sigmoid()
        )

        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'activation_fn'):
                if m.activation_fn == 'tanh':
                    nn.init.xavier_normal_(m.weight.data)
                elif m.activation_fn == 'relu':
                    nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            else:
                nn.init.xavier_normal_(m.weight.data, gain=1.0)

    def forward(self, input):
        return self.main(input)
