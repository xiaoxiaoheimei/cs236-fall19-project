'''
the trainer for the homomorphic interpolation model
'''
from __future__ import print_function
import torch
from torch import nn
from optimizer.base_optimizer import base_optimizer
from network import base_network
from network.loss import classification_loss_list, perceptural_loss
from util import util, logger, opt, curves
import os
import util.tensorWriter as vs
from collections import OrderedDict
import tensorboardX

import pdb 

log = logger.logger()


class optimizer(base_optimizer):

    def __init__(self, model, option=opt.opt()):
        super(optimizer, self).__init__()
        self._default_opt()
        self.opt.merge_opt(option)
        self._get_model(model)
        self._get_aux_nets()
        self._define_optim()
        # self.writer = curves.writer(log_dir=self.opt.save_dir + '/log')
        util.mkdir(self.opt.save_dir+'/log')
        self.writer = tensorboardX.SummaryWriter(log_dir=self.opt.save_dir + '/log')
        if self.opt.continue_train:
            self.load()

    def _default_opt(self):
        self.opt = opt.opt()
        self.opt.save_dir = '/checkpoints/default'
        self.opt.n_discrim = 5

    def set_input(self, input):
        self.image, self.attribute = input
        self.image = util.toVariable(self.image).cuda()
        self.attribute = [util.toVariable(att.cuda()) for att in self.attribute]

    def zero_grad(self):
        self.encoders.zero_grad()
        self.interp_nets.zero_grad()
        self.decoders.zero_grad()
        self.discrims.zero_grad()
        self.KGTransforms.zero_grad()

    def _get_aux_nets(self):
        self.vgg_teacher = nn.DataParallel(base_network.VGG(pretrained=True)).cuda()
        self.perceptural_loss = perceptural_loss().cuda()
        self.KGTransform = nn.DataParallel(nn.Conv2d(512, 512, 1)).cuda()

    def _get_model(self, model, lw):
        """
        Args: 
          @model (list of ModuleList): [encoders, interp_nets, decoders, discrims]
          @lw (list of float): weight of each stacked GAN structure
        """
        encoders, interp_nets, decoders, discrims = model
        self.depth = len(encoders)
        self.lw = lw
        #put each module to GPU
        self.encoders = encoders.cuda()
        self.interp_nets = interp_nets.cuda()
        self.decoders = decoders.cuda()
        self.discrims = discrims.cuda()
        with open(self.opt.save_dir + '/encoders.txt', 'w') as f:
            print(encoders, file=f)
        with open(self.opt.save_dir + '/interp_nets.txt', 'w') as f:
            print(interp_nets, file=f)
        with open(self.opt.save_dir + '/decoders.txt', 'w') as f:
            print(decoders, file=f)
        with open(self.opt.save_dir + '/discrims.txt', 'w') as f:
            print(discrims, file=f)

    def _define_optim(self):
        self.optim_encoders = torch.optim.Adam(self.encoders.parameters(), lr=1e-4, betas=[0.5, 0.999])
        self.optim_interps = torch.optim.Adam(self.interp_nets.parameters(), lr=1e-4, betas=[0.5, 0.999])
        self.optim_decoders = torch.optim.Adam(self.decoders.parameters(), lr=1e-4)
        self.optim_discrims = torch.optim.Adam(self.discrims.parameters(), lr=1e-4, betas=[0.5, 0.999])
        self.optim_KGTransforms = torch.optim.Adam(self.KGTransform.parameters(), lr=1e-4)

    def load(self, label='latest'):
        save_dir = self.opt.save_dir + '/{}-{}.pth'
        self._check_and_load(self.encoders, save_dir.format('encoders', label))
        self._check_and_load(self.interp_nets, save_dir.format('interp_nets', label))
        self._check_and_load(self.decoders, save_dir.format('decoders', label))
        self._check_and_load(self.discrims, save_dir.format('discrims', label))
        self._check_and_load(self.KGTransforms, save_dir.format('KGTransforms', label))

    def save(self, label='latest'):
        save_dir = self.opt.save_dir + '/{}-{}.pth'
        torch.save(self.encoders.state_dict(), save_dir.format('encoder', label))
        torch.save(self.interp_nets.state_dict(), save_dir.format('interp_net', label))
        torch.save(self.decoders.state_dict(), save_dir.format('decoder', label))
        torch.save(self.discrims.state_dict(), save_dir.format('discrim', label))
        torch.save(self.KGTransforms.state_dict(), save_dir.format('KGTransform', label))

    def optimize_parameters(self, global_step):
        pdb.set_trace()
        self.encoders.train()
        self.interp_nets.train()
        self.discrims.train()
        self.decoders.train()
        self.KGTransforms.train()
        self.loss = OrderedDict()
        ''' define v '''
        self.v = util.toVariable(self.generate_select_vector()).cuda() #(batch, branch_num)
        self.rand_idx = torch.randperm(self.image.size(0)).cuda() #(batch,)
        self.image_permute = self.image[self.rand_idx]
        self.attr_permute = []
        for att in self.attribute:
            self.attr_permute += [att[self.rand_idx]]
        ''' compute the target attributes '''
        self.attr_interp = []
        for i, (att, attp) in enumerate(zip(self.attribute, self.attr_permute)):
            self.attr_interp += [att + self.v[:, i:i + 1] * (attp - att)] #use permuted images as the target, v=0/v=1 corresponding to the source/target images respectively.
        ''' pre-computed variables '''
        #compute latent representation for each stacked layer
        self.feat = self.stacked_encoder(self.image, self.depth)
        self.feat_permute = []
        for feat in self.feat:
            self.feat_permute.append(feat[self.rand_idx])
        self.feat_interp = []
        for feat, feat_permute, interp_net in zip(self.feat, self.feat_permute, self.interp_nets):
            stack_interp = interp_net(feat, feat_permute, self.v)
            self.feat_interp.append(stack_interp)

        self.zero_grad()
        self.compute_dec_loss().backward(retain_graph=True)
        self.optim_decoder.step()

        self.zero_grad()
        self.compute_discrim_loss().backward(retain_graph=True)
        self.optim_discrim.step()

        self.zero_grad()
        self.compute_KGTransform_loss().backward(retain_graph=True)
        self.optim_KGTransform.step()

        if global_step % self.opt.n_discrim == 0:
            self.zero_grad()
            self.compute_enc_int_loss().backward()
            self.optim_encoder.step()
            self.optim_interp.step()

    def stacked_encoder(self, image, k):
        """
        Args: 
           image (tensor): image input tensor. (batch, channel, width, height)
           k (int): id of the latent space, start from 0
        Return:
           feats (tensor list): latent representations of the latent spaces 0 to k, [r_{0}, ... , r_{k}]
        """
        enc_feats = [image]
        for encoder in self.encoders[0:k+1]:
            f = encoder(feats[-1]) 
            enc_feats.append(f)
        enc_feats = enc_feats[1:]
        return enc_feats

    def stacked_decoder(self, feat, k):
        """
        Args:
           k (int): id of the latent space, start from 0
           feat (tensor): latent representation of the kth latent space
        Return:
           dec_feats (tensor list): decoded results through out the stack. [s_{0} (image space), s_{1}, ..., s_{k-1}]
        """
        dec_feats = [feat]
        for decoder in reversed(self.decoders[0:k+1]):
            f = decoder(dec_feats[-1])
            dec_feats.append(f)
        dec_feats = dec_feats[:0:-1]
        return dec_feats

    def compute_dec_loss(self):
        self.loss['dec'] = 0
        for i, feat in enumerate(self.feat):
            #decode latent feature through stacked GAN
            im_out = self.stacked_decoder(feat, i)[0]
            self.loss[f'dec_per_stack{i}']  = self.perceptural_loss(im_out, self.image) #the perceptural loss of the ith GAN
            self.loss['dec'] += self.loss[f'dec_per_stack{i}'] * self.lw[i]
            self.loss[f'dec_mse_stack{i}'] = nn.MSELoss()(im_out, self.image.detach()) #why detach here
            self.loss['dec'] += self.loss[f'dec_mse_stack{i}'] * self.lw[i]
        return self.loss['dec']

    def compute_discrim_loss(self):
        self.loss['discrim'] = 0
        for i, discrim, feat, feat_interp in enumerate(zip(self.discrims, self.feat, self.feat_interp)):
            discrim_real, real_attr = discrim(feat.detach())
            discrim_interp, interp_attr = discrim(feat_interp.detach())
            ''' gradient penality '''
            gp_interpolate = self.random_interpolate(feat.data, feat_interp.data)
            gp_interpolate = util.toVariable(gp_interpolate, requires_grad=True)
            discrim_gp_interpolated, _ = discrim(gp_interpolate)
            self.loss[f'discrim_gp_stack{i}'] = util.gradient_penalty(gp_interpolate, discrim_gp_interpolated) * 100.
            self.loss['discrim'] += self.loss[f'discrim_gp_stack{i}'] * self.lw[i]
            ''' the GAN loss '''
            self.loss[f'discrim_gan_stack{i}'] = (discrim_interp - discrim_real).mean()
            self.loss['discrim'] += self.loss[f'discrim_gan_stack{i}'] * self.lw[i]
            ''' the attribute classification loss '''
            att_detach = [att.detach() for att in self.attribute]
            self.loss[f'discrim_cls_stack{i}'] = classification_loss_list(interp_attr, att_detach) #Rigorous Training in each latent space
            self.loss['discrim'] += self.loss[f'discrim_cls_stack{i}'] * self.lw[i]
            return self.loss['discrim']

    def compute_KGTransform_loss(self):
        self.loss['KGTransform'] = 0
        feat_T1 = self.vgg_teacher(self.image)[-1]
        for feat, KGTransform in zip(self.feat, self.KGTransforms):
            feat_T2 = self.KGTransform(feat)
            self.loss[f'KGTransform_stack{i}'] = nn.MSELoss()(feat_T2, feat_T1.detach())
            self.loss['KGTransform'] += self.loss[f'KGTransform_stack{i}'] * self.lw[i]
        return self.loss['KGTransform']

    def compute_enc_int_loss(self):
        self.loss['enc_int'] = 0
        # discrim_real, out_attr = self.discrim(self.feat)
        for i, feat, feat_permute, feat_interp in enumerate(zip(self.feat, self.feat_permute, self.feat_interp)):
            discrim_interp, interp_attr = discrim[i](feat_interp)
            ''' GAN loss '''
            self.loss[f'enc_int_gan_stack{i}'] = -discrim_interp.mean()
            self.loss['enc_int'] += self.loss[f'enc_int_gan_stack{i}'] * self.lw[i]
            ''' classification loss '''
            interp_detach = [att.detach() for att in self.attr_interp]
            self.loss[f'enc_int_cls_stack{i}'] = classification_loss_list(interp_attr, interp_detach) #Try to keep Homographic property in each latent space.
            self.loss['enc_int'] += self.loss[f'enc_int_cls_stack{i}'] * self.lw[i]
            ''' interp all loss '''
            feat_interp_all = self.interp_nets[i](feat.detach(), feat_permute.detach(),
                                                       self.generate_select_vector(type='select_all'))
            self.loss[f'enc_int_all_stack{i}'] = nn.MSELoss()(feat_interp_all, feat_permute.detach())
            self.loss['enc_int'] += self.loss[f'enc_int_all_stack{i}'] * self.lw[i]
            # feat_interp_none = self.interp_net(self.feat.detach(), self.feat_permute.detach(),
            #                                   self.generate_select_vector(type='select_none'))
            # self.loss['enc_int_none'] = nn.MSELoss()(feat_interp_none, self.feat.detach())
            # self.loss['enc_int'] += self.loss['enc_int_none']
            ''' reconstruction loss '''
            feat_t = feat
            for stack_decoder in reversed(self.decoders[0:i+1]):
                im_out = stack_decoder(feat_t)
                feat_t = im_out
            loss[f'enc_int_mse_stack{i}'] = nn.MSELoss()(im_out, self.image.detach())
            self.loss['enc_int'] += self.loss[f'enc_int_mse_stack{i}'] * self.lw[i]
            self.loss['enc_int_per_stack{i}'] = self.perceptural_loss(im_out, self.image.detach())
            self.loss['enc_int'] += self.loss[f'enc_int_per_stack{i}'] * self.lw[i]
            ''' knowledge guidance loss '''
            feat_T1 = self.vgg_teacher(self.image)[-1]
            feat_T2 = self.KGTransforms[i](feat)
            self.loss[f'enc_int_KG_stack{i}'] = nn.MSELoss()(feat_T2, feat_T1.detach())
            self.loss['enc_int'] += self.loss[f'enc_int_KG_stack{i}'] * self.lw[i]
        return self.loss['enc_int']

    def get_current_errors(self):
        return self.loss

    def stacked_interp_test(self, img1, img2):
        """
        Interp in kth latent space
        Args:
          img1, img2 (tensor): img1@soure, img2@target, (channel, height, width)
          k (int): id of the interpolated latent space.
        Return:
          result_full_stack (list of tensor): each element is the interpolated result in a specific latent space.
        """
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        stacked_feat1 = self.stacked_encoder(img1) #list of self.depth elements
        stacked_feat2 = self.stacked_encoder(img2)
        result_full_stack = []
        for i, interp_net, fea1, fea2 in enumerate(zip(self.interp_nets, stacked_feat1, stacked_fea2)):
            n_branches = interp_net.module.n_branch
            result_per_stack = []
            for attr_idx in range(n_branches):
                result_branch = [img1.data.cpu()]
                for strength in [0, 0.5, 1]:
                    attr_vec = torch.zeros(1, n_branches + 1)
                    attr_vec[:, attr_idx] = strength
                    attr_vec = util.toVariable(attr_vec).cuda()
                    interp_feat = interp_net(feat1, feat2, attr_vec)
                    out_tmp = self.stacked_decoder(interp_feat, i)[0]
                    result_branch += [out_tmp.data.cpu()]
                result_branch += [img2.data.cpu()]
                result_branch = torch.cat(result_branch, dim=3)
                result_per_stack += [result_branch]
            result_branch = [img1.data.cpu()]
            # interpolate all the attributes
            for strength in [0, 0.5, 1]:
                attr_vec = torch.ones(1, n_branches) * strength
                attr_vec = util.toVariable(attr_vec).cuda()
                interp_feat = interp_net(feat1, feat2, attr_vec)
                out_tmp = self.stacked_decoder(interp_feat, i)[0]
                result_branch += [out_tmp.data.cpu()]
            result_branch += [img2.data.cpu()]
            result_branch = torch.cat(result_branch, dim=3)
            result_per_stack += [result_branch]
            result_per_stack = torch.cat(result_per_stack, dim=2)
            result_full_stack += [result_per_stack]
        #result_full_stack = torch.cat(result_full_stack, dim=1)
        return result_full_stack

    def interp_test(self, img1, img2):
        '''
        testing the interpolation effect.
        :param type: "single" and "accumulate"
        :return: a torch image that combines the interpolation results
        '''
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)
        result_map = []
        n_branches = self.interp_net.module.n_branch
        for attr_idx in range(n_branches):
            result_row = [img1.data.cpu()]
            for strength in [0, 0.5, 1]:
                attr_vec = torch.zeros(1, n_branches + 1)
                attr_vec[:, attr_idx] = strength
                attr_vec = util.toVariable(attr_vec).cuda()
                interp_feat = self.interp_net(feat1, feat2, attr_vec)
                out_tmp = self.decoder(interp_feat)
                result_row += [out_tmp.data.cpu()]
            result_row += [img2.data.cpu()]
            result_row = torch.cat(result_row, dim=3)
            result_map += [result_row]
        result_row = [img1.data.cpu()]
        # interpolate all the attributes
        for strength in [0, 0.5, 1]:
            attr_vec = torch.ones(1, n_branches) * strength
            attr_vec = util.toVariable(attr_vec).cuda()
            interp_feat = self.interp_net(feat1, feat2, attr_vec)
            out_tmp = self.decoder(interp_feat)
            result_row += [out_tmp.data.cpu()]
        result_row += [img2.data.cpu()]
        result_row = torch.cat(result_row, dim=3)
        result_map += [result_row]

        result_map = torch.cat(result_map, dim=2)
        return result_map

    def stacked_save_samples(self, global_step=0):
        self.encoders.eval()
        self.interp_nets.eval()
        self.discrims.eval()
        self.decoders.eval()
        n_pairs = 20
        save_path_single = [os.path.join(self.opt.save_dir, f'samples/single_depth{i}') for i in range(self.depth)]
        for path in save_path_single:
            util.mkdir(path)
        map_single = []
        n_branches = self.interp_nets[0].module.n_branch
        for i in range(n_pairs):
            img1, _ = self.test_dataset[i]
            img2, _ = self.test_dataset[i + 1]
            img1 = util.toVariable(img1).cuda()
            img2 = util.toVariable(img2).cuda()
            map_single += [self.stacked_interp_test(img1, img2)]
        for i, stacked_map_single in enumerate(zip(*map_single)):
            stacked_single_seq = torch.cat(stacked_map_single, dim=0)
            stacked_single_seq = vs.untransformTensor(stacked_single_seq)
            vs.writeTensor(os.path.join(save_path_single[i], '%d.jpg' % global_step), stacked_single_seq, nRow=2)
        ##################################################
        save_path = [os.path.join(self.opt.save_dir, f'interp_curve_depth{i}') for i in range(self.depth)]
        for path in save_path:
            util.mkdir(path)
        for k, interp_net in enumerate(self.interp_nets):
            im_out = [self.image.data.cpu()]
            v = torch.zeros(self.image.size(0), n_branches)
            v = util.toVariable(v).cuda()
            feat = interp_net(self.feat[k], self.feat_permute[k], v)
            out_now = self.stacked_decoder(feat, k)[0]
            im_out += [out_now.data.cpu()]
            for i in range(n_branches):
                log(i)
                v = torch.zeros(self.image.size(0), n_branches)
                v[:, 0:i + 1] = 1
                v = util.toVariable(v).cuda()
                feat = interp_net(self.feat[k].detach(), self.feat_permute[k].detach(), v)
                out_now = self.stacked_decoder(feat.detach(), k)[0]
                im_out += [out_now.data.cpu()]
            im_out += [self.image_permute.data.cpu()]
            im_out = [util.toVariable(tmp) for tmp in im_out]
            im_out = torch.cat(im_out, dim=0)
            im_out = vs.untransformTensor(im_out.data.cpu())
            vs.writeTensor('%s/%d.jpg' % (save_path[k], global_step), im_out, nRow=self.image.size(0))

    def save_samples(self, global_step=0):
        self.encoder.eval()
        self.interp_net.eval()
        self.discrim.eval()
        self.decoder.eval()
        n_pairs = 20
        save_path_single = os.path.join(self.opt.save_dir, 'samples/single')
        util.mkdir(save_path_single)
        map_single = []
        n_branches = self.interp_net.module.n_branch
        for i in range(n_pairs):
            img1, _ = self.test_dataset[i]
            img2, _ = self.test_dataset[i + 1]
            img1 = util.toVariable(img1).cuda()
            img2 = util.toVariable(img2).cuda()
            map_single += [self.interp_test(img1, img2)]
        map_single = torch.cat(map_single, dim=0)
        map_single = vs.untransformTensor(map_single)
        vs.writeTensor(os.path.join(save_path_single, '%d.jpg' % global_step), map_single, nRow=2)
        ##################################################
        save_path = os.path.join(self.opt.save_dir, 'interp_curve')
        util.mkdir(save_path)
        im_out = [self.image.data.cpu()]
        v = torch.zeros(self.image.size(0), n_branches)
        v = util.toVariable(v).cuda()
        feat = self.interp_net(self.feat, self.feat_permute, v)
        out_now = self.decoder(feat)
        im_out += [out_now.data.cpu()]
        for i in range(n_branches):
            log(i)
            v = torch.zeros(self.image.size(0), n_branches)
            v[:, 0:i + 1] = 1
            v = util.toVariable(v).cuda()
            feat = self.interp_net(self.feat.detach(), self.feat_permute.detach(), v)
            out_now = self.decoder(feat.detach())
            im_out += [out_now.data.cpu()]
        im_out += [self.image_permute.data.cpu()]
        im_out = [util.toVariable(tmp) for tmp in im_out]
        im_out = torch.cat(im_out, dim=0)
        im_out = vs.untransformTensor(im_out.data.cpu())
        vs.writeTensor('%s/%d.jpg' % (save_path, global_step), im_out, nRow=self.image.size(0))

    def _generate_select_vector(self, n_branches, type='uniform'):
        '''
        generate the select vector to select the interpolation curve
        type:

        :return: nSample x selct_dims, which indicates which attribute to be transferred.
        '''

        if type == 'one_attr_randsample':  # each sample has one random selected attribute
            selected_vector = []
            for i in range(self.image.size(0)):
                tmp = torch.randperm(n_branches)[0]  # randomly select one attribute
                # log('generate_select_vector: tmp:', tmp)
                one_hot_vec = torch.zeros(1, n_branches)
                one_hot_vec[:, tmp] = 1
                # log('generate_select_vector: one_hot_vec:', one_hot_vec)
                selected_vector += [one_hot_vec]
            selected_vector = torch.cat(selected_vector, dim=0)
            # log('one-attr-randsample', selected_vector)
            selected_vector = util.toVariable(selected_vector).cuda()
            return selected_vector
        elif type == 'one_attr_batch':  # each batch has one common selected attribute
            raise NotImplemented
        elif type == 'uniform':
            selected_vector = torch.rand(self.image.size(0), n_branches)
            selected_vector = util.toVariable(selected_vector).cuda()
            return selected_vector
        elif type == 'uniform_binarize':
            selected_vector = torch.rand(self.image.size(0), n_branches)
            selected_vector = (selected_vector > 0.5).float() * 1.
            selected_vector = util.toVariable(selected_vector).cuda()
            return selected_vector
        elif type == 'select_all':
            selected_vector = torch.ones(self.image.size(0), n_branches)
            selected_vector = util.toVariable(selected_vector).cuda()
            return selected_vector
        elif type == 'select_none':
            selected_vector = torch.zeros(self.image.size(0), n_branches)
            # log('selective_none', torch.sum(selected_vector))
            selected_vector = util.toVariable(selected_vector).cuda()
            return selected_vector
        else:
            raise NotImplemented

    def generate_select_vector(self, type='uniform'):
        n_branches = self.interp_net.module.n_branch
        return self._generate_select_vector(n_branches, type)

    def random_interpolate(self, gt, pred):
        batch_size = gt.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).cuda()
        # alpha = alpha.expand(gt.size()).cuda()
        interpolated = gt * alpha + pred * (1 - alpha)
        return interpolated
