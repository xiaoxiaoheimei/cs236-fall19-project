import torch
from torch import nn
from network import resnet50

class discriminator(nn.Module):
    def __init__(self, in_channels=3, wgan=True):
        super(discriminator, self).__init__()
        self.wgan = wgan
        self.model = nn.Sequential(*[
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)])
        self.fn = nn.Linear(256*6*6, 1)

    def forward(self, x):
        # print('input', x.shape)
        bs = x.shape[0]
        x = self.model(x)
        if self.wgan == True:
            x = self.fn(x.view(bs,-1))
        return x


class decoder_face_rotate(nn.Module):
    def __init__(self):
        super(decoder_face_rotate, self).__init__()
        self.lm_encoder = nn.Sequential(*[
            nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()])

        self.model = nn.Sequential(*[
            nn.Conv2d(2048+256, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            res_block(in_channels=512, out_channels=512),
            res_block(in_channels=512, out_channels=512),
            res_block(in_channels=512, out_channels=512),
            res_block(in_channels=512, out_channels=256),
            deconv_block(in_channels=256),
            res_block(in_channels=256, out_channels=128),
            deconv_block(in_channels=128),
            res_block(in_channels=128, out_channels=64),
            deconv_block(in_channels=64),
            res_block(in_channels=64, out_channels=32),
            deconv_block(in_channels=32),
            res_block(in_channels=32, out_channels=32),
            deconv_block(in_channels=32),
            nn.Conv2d(32, 3, kernel_size=1, padding=0),
            # nn.Tanh(),
        ])

    def forward(self, feat_id, lm):
        feat_lm = self.lm_encoder(lm)
        feat = torch.cat([feat_id, feat_lm], dim=1)
        y = self.model(feat)
        # return (y + 1) * 0.5
        return y

class decoder_face_normalize(nn.Module):
    def __init__(self):
        super(decoder_face_normalize, self).__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(2048, 512, kernel_size=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            res_block(in_channels=512, out_channels=512),
            res_block(in_channels=512, out_channels=512),
            res_block(in_channels=512, out_channels=512),
            res_block(in_channels=512, out_channels=256),
            deconv_block(in_channels=256),
            res_block(in_channels=256, out_channels=128),
            deconv_block(in_channels=128),
            res_block(in_channels=128, out_channels=64),
            deconv_block(in_channels=64),
            res_block(in_channels=64, out_channels=32),
            deconv_block(in_channels=32),
            res_block(in_channels=32, out_channels=32),
            deconv_block(in_channels=32),
            nn.Conv2d(32, 3, kernel_size=1, padding=0),
            nn.Tanh(),
        ])

    def forward(self, feat_id):
        y = self.model(feat_id)
        return (y + 1) * 0.5
        # return y


class res_block(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(res_block, self).__init__()
        self.model = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ])
    def forward(self, x):
        y = self.model(x)
        return y

class deconv_block(nn.Module):
    def __init__(self, in_channels=256, upsample_mode='nearest'):
        super(deconv_block, self).__init__()
        self.model = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        ])
    def forward(self, x):
        y = self.model(x)
        return y
