'''
the trainer for the homomorphic interpolation model
'''
from __future__ import print_function
import torch
from torch import nn
import torch.nn.functional as F
from optimizer.base_optimizer import base_optimizer
from network import base_network
from network.loss import classification_loss_list, perceptural_loss
from util import util, logger, opt, curves
import os
import util.tensorWriter as vs
from collections import OrderedDict
import tensorboardX

log = logger.logger()


class optimizer(base_optimizer):

    def __init__(self, model, option=opt.opt()):
        super(optimizer, self).__init__()
        self._default_opt()
        self.opt.merge_opt(option)
        self._get_model(model)
        self._define_optim()
        # self.writer = curves.writer(log_dir=self.opt.save_dir + '/log')
        util.mkdir(self.opt.save_dir+'/log')
        self.writer = tensorboardX.SummaryWriter(log_dir=self.opt.save_dir + '/log')
        self.load_pretrained(weights_path='checkpoints/resnet50_face/resnet50_ft_dims_2048.pth')
        self.opt.continue_train = False
        if self.opt.continue_train:
            self.load()

    def _default_opt(self):
        self.opt = opt.opt()
        self.opt.save_dir = 'checkpoints/face_normalize'
        util.mkdir(self.opt.save_dir)

    def set_input(self, input):
        self.image_normal, self.image_tilt = input
        self.batch_size = self.image_normal.shape[0]
        self.mean = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).expand_as(self.image_normal).cuda()
        self.stdv = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).expand_as(self.image_normal).cuda()
        # pretrained VGG-face input 0~225
        self.image_normal = util.toVariable(255*self.image_normal).cuda()
        self.image_tilt = util.toVariable(255*self.image_tilt).cuda()
        # self.landmark_trans = util.toVariable(self.landmark_trans).cuda()
        # self.landmark_same = util.toVariable(self.landmark_same).cuda()
        # self.attribute = [util.toVariable(att.cuda()) for att in self.attribute]

    def zero_grad(self):
        # self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.discrim_rf.zero_grad()
        # self.discrim_lm.zero_grad()

    def _get_model(self, model):
        encoder, decoder, discrim_rf = model
        self.encoder = encoder.cuda()
        self.decoder = decoder.cuda()
        self.discrim_rf = discrim_rf.cuda()
        # self.discrim_lm = discrim_lm.cuda()
        with open(self.opt.save_dir + '/encoder.txt', 'w') as f:
            print(encoder, file=f)
        with open(self.opt.save_dir + '/decoder.txt', 'w') as f:
            print(decoder, file=f)
        with open(self.opt.save_dir + '/discrim_rf.txt', 'w') as f:
            print(discrim_rf, file=f)
        # with open(self.opt.save_dir + '/discrim_lm.txt', 'w') as f:
        #     print(discrim_lm, file=f)

    def _define_optim(self):
        # self.optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=1e-4, betas=[0.5, 0.999])
        self.optim_decoder = torch.optim.Adam(self.decoder.parameters(), lr=1e-4)
        self.optim_discrim_rf = torch.optim.Adam(self.discrim_rf.parameters(), lr=1e-4)
        # self.optim_discrim_lm = torch.optim.Adam(self.discrim_lm.parameters(), lr=1e-4)

    def load(self, label='latest'):
        save_dir = self.opt.save_dir + '/{}-{}.pth'
        self._check_and_load(self.decoder, save_dir.format('decoder', label))
        self._check_and_load(self.discrim_rf, save_dir.format('discrim_rf', label))
        # self._check_and_load(self.discrim_lm, save_dir.format('discrim_lm', label))

    def load_pretrained(self, weights_path='checkpoints/resnet50_face/resnet50_ft_dims_2048.pth'):
        state_dict = torch.load(weights_path)
        self.encoder.load_state_dict(state_dict, strict=False)
        # self.encoder.load_state_dict(state_dict)

    def save(self, label='latest'):
        save_dir = self.opt.save_dir + '/{}-{}.pth'
        # torch.save(self.encoder.state_dict(), save_dir.format('encoder', label))
        torch.save(self.decoder.state_dict(), save_dir.format('decoder', label))
        torch.save(self.discrim_rf.state_dict(), save_dir.format('discrim_rf', label))
        # torch.save(self.discrim_lm.state_dict(), save_dir.format('discrim_lm', label))

    def optimize_parameters(self, global_step):
        self.encoder.eval()
        self.decoder.train()
        self.discrim_rf.train()
        # self.discrim_lm.train()
        self.loss = OrderedDict()
        ''' define features from pretrained VGG-face '''
        # input (0-225)
        _, _, self.feat_normal = self.encoder(self.image_normal.detach())
        _, _, self.feat_tilt = self.encoder(self.image_tilt.detach())

        ''' define output from decoder '''
        self.output_normal = self.decoder(self.feat_normal.detach())
        self.output_tilt = self.decoder(self.feat_tilt.detach())
        self.output_normal_norm = (self.output_normal - self.mean) / self.stdv
        self.output_tilt_norm = (self.output_tilt - self.mean) / self.stdv
        # self.output_trans = self.decoder(self.feat_identity.detach(), self.landmark_trans.detach())
        # self.output_same = self.decoder(self.feat_identity.detach(), self.landmark_same.detach())

        # Pa = self.landmark_trans[:,0:1,:,:] / 255.
        # Pb = self.landmark_trans[:,1:2,:,:] / 255.
        # self.IAPb = torch.cat([self.output_same, Pb], dim=1) # Ia', Pb
        # self.IaPb = torch.cat([self.image / 255., Pb], dim=1) # Ia, Pb
        # self.IBPb = torch.cat([self.output_trans, Pb], dim=1) # Ib', Pb
        # self.IBPa = torch.cat([self.output_trans, Pa], dim=1) # Ib', Pa
        # self.IaPa = torch.cat([self.image / 255., Pa], dim=1) # Ia, Pa

        ''' define losses'''
        # self.zero_grad()
        # self.compute_discrim_lm_loss().backward(retain_graph=True)
        # self.optim_discrim_lm.step()

        self.zero_grad()
        self.compute_discrim_rf_loss().backward(retain_graph=True)
        self.optim_discrim_rf.step()

        self.zero_grad()
        self.compute_dec_loss().backward(retain_graph=True)
        self.optim_decoder.step()

    def compute_dec_loss(self):
        self.loss['dec'] = 0
        ''' mse between Ia, Ia' '''
        self.loss['dec_mse_image'] = nn.MSELoss()(self.output_normal_norm, self.image_normal.detach()/255.)
        self.loss['dec'] += self.loss['dec_mse_image']

        ''' mse between features from VGG-face (id, Fa'), (id, Fb') '''
        _, _, self.feat_output_normal = self.encoder(255.*self.output_normal_norm)
        _, _, self.feat_output_tilt = self.encoder(255.*self.output_tilt_norm)
        self.loss['dec_mse_id_normal'] = nn.MSELoss()(self.feat_output_normal, self.feat_normal.detach())
        self.loss['dec'] += 0.1 * self.loss['dec_mse_id_normal']
        self.loss['dec_mse_id_tilt'] = nn.MSELoss()(self.feat_output_tilt, self.feat_tilt.detach())
        self.loss['dec'] += 0.1 * self.loss['dec_mse_id_tilt']

        ''' normal GAN loss '''
        pred_fake_normal = self.discrim_rf(self.output_normal_norm)
        pred_fake_tilt = self.discrim_rf(self.output_tilt_norm)
        self.loss['dec_gan_rf'] = -0.5 * (F.logsigmoid(pred_fake_normal) + F.logsigmoid(pred_fake_tilt)).mean()
        self.loss['dec'] += self.loss['dec_gan_rf']

        # pred_lm_fake_1 = self.discrim_lm(self.IAPb)
        # pred_lm_fake_2 = self.discrim_lm(self.IaPb)
        # pred_lm_fake_3 = self.discrim_lm(self.IBPb)
        # pred_lm_fake_4 = self.discrim_lm(self.IBPa)
        # self.loss['dec_gan_lm'] = -0.25 * (F.logsigmoid(pred_lm_fake_1) + F.logsigmoid(pred_lm_fake_2) + \
        #                             F.logsigmoid(pred_lm_fake_3) + F.logsigmoid(pred_lm_fake_4)).mean()
        # self.loss['dec'] += self.loss['dec_gan_lm']

        return self.loss['dec']

    def compute_discrim_rf_loss(self):
        ''' define output from discrim real/fake'''
        pred_fake_normal = self.discrim_rf(self.output_normal_norm.detach())
        pred_fake_tilt = self.discrim_rf(self.output_tilt_norm.detach())
        pred_rf_real = self.discrim_rf(self.image_normal.detach()/255.)

        self.loss['dis_rf'] = 0
        ''' normal GAN loss '''
        self.loss['dis_rf_gan'] = -(0.5*(F.logsigmoid(-pred_fake_normal) + F.logsigmoid(-pred_fake_tilt)) + \
                                    F.logsigmoid(pred_rf_real)).mean()
        self.loss['dis_rf'] += self.loss['dis_rf_gan']

        return self.loss['dis_rf']

    # def compute_discrim_lm_loss(self):
    #     ''' define output from discrim landmarks'''
    #     pred_lm_fake_1 = self.discrim_lm(self.IAPb.detach())
    #     pred_lm_fake_2 = self.discrim_lm(self.IaPb.detach())
    #     pred_lm_fake_3 = self.discrim_lm(self.IBPb.detach())
    #     pred_lm_fake_4 = self.discrim_lm(self.IBPa.detach())
    #     pred_lm_real = self.discrim_lm(self.IaPa.detach())
    #
    #     self.loss['dis_lm'] = 0
    #     ''' normal GAN loss '''
    #     self.loss['dis_lm_gan'] = -(0.25*(F.logsigmoid(-pred_lm_fake_1) + F.logsigmoid(-pred_lm_fake_2) + \
    #                                 F.logsigmoid(-pred_lm_fake_3) + F.logsigmoid(-pred_lm_fake_4)) + \
    #                                 F.logsigmoid(pred_lm_real)).mean()
    #     self.loss['dis_lm'] += self.loss['dis_lm_gan']
    #
    #     return self.loss['dis_lm']

    def get_current_errors(self):
        return self.loss

    def save_samples(self, test_dataloader, global_step=0):
        self.encoder.eval()
        self.decoder.eval()
        self.discrim_rf.eval()
        # self.discrim_lm.eval()

        save_path_single = os.path.join(self.opt.save_dir, 'samples/test')
        util.mkdir(save_path_single)
        for i, data in enumerate(test_dataloader):
            image_normal, image_tilt = data
            _, _, feat_normal = self.encoder(255.*image_normal)
            _, _, feat_tilt = self.encoder(255.*image_tilt)
            # output normalized around mean / std
            output_normal = self.decoder(feat_normal)
            output_tilt = self.decoder(feat_tilt)
            mean = torch.Tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).expand_as(image_normal).cuda()
            stdv = torch.Tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).expand_as(image_normal).cuda()
            output_normal_norm = (output_normal - mean) / stdv
            output_tilt_norm = (output_tilt - mean) / stdv
            output = torch.cat([output_normal_norm, util.toTensor(image_normal).cuda(),
                                output_tilt_norm, util.toTensor(image_tilt).cuda()], dim=-1)
            break
        output = vs.untransformTensor(output.detach())
        vs.writeTensor_2(os.path.join(save_path_single, '%d.jpg' % global_step), output)
