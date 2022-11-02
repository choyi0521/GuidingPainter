import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, lr_scheduler
import pytorch_lightning as pl
from modules.wnet_generator import WNetGenerator
from modules.nlayer_discriminator import NLayerDiscriminator
from modules.losses import GANLoss
from utils import rgb2image, calculate_psnr_torch, init_weights
from color_map import get_color_map
from torch.distributions.geometric import Geometric
import os


class AskModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        hint_channels = args.max_hints if args.feed_segments else 0
        self.generator = WNetGenerator(
            args.input_channels,
            args.max_hints,
            args.input_channels + args.output_channels + hint_channels + 1,
            args.output_channels,
            args.use_bilinear
        )
        self.discriminator = NLayerDiscriminator(args.input_channels + args.output_channels)

        if args.init_type is not None:
            init_weights(self.generator, args.init_type)
            init_weights(self.discriminator, args.init_type)

        # losses
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.gan_loss = GANLoss('lsgan')

    def forward(self, base, real, tau=1.0, deterministic=False):
        logits, seg = self.segment(base, tau, deterministic)
        
        fcolor, fmask = self._get_full_hint(seg, real)
        act = self._get_random_act((base.shape[0], self.args.max_hints), self.args.stop_prob)

        color, mask = self._apply_act(fcolor, fmask, act)
        pred = self.color(base, color, mask, seg)

        return logits, fcolor, fmask, color, mask, pred

    def segment(self, base, tau=1.0, deterministic=False):
        logits, seg = self.generator.segment(base, tau, deterministic)
        return logits, seg
    
    def color(self, base, color, mask, seg):
        if self.args.feed_segments:
            pred = self.generator(torch.cat((base, color.sum(dim=1), mask.sum(dim=1) - self.args.ask_mask_cent, seg - self.args.ask_mask_cent), dim=1))
        else:
            pred = self.generator(torch.cat((base, color.sum(dim=1), mask.sum(dim=1) - self.args.ask_mask_cent), dim=1))
        return pred
    
    def _apply_act(self, fcolor, fmask, act):
        act = act.view(*act.shape, 1, 1, 1)
        color = fcolor * act
        mask = fmask * act
        return color, mask
    
    def _get_random_act(self, shape, stop_prob):
        act = torch.zeros(shape, device=self.device)
        live = torch.bernoulli(torch.full(shape, 1.0-stop_prob, device=self.device))
        for i in range(shape[1]):
            if i == 0:
                act[:, 0] = live[:, 0]
            else:
                act[:, i] = act[:, i-1] * live[:, i]
        return act
    
    def _get_full_hint(self, seg, real, eps=1e-20):
        """
        seg: batch_size x max_hints x height x width
        real: batch_size x color_channels x height x width
        act: batch_size x max_hints
        fcolor: batch_size x max_hints x color_channels x height x width
        fmask: batch_size x max_hints x 1 x height x width
        """
        fmask = seg.unsqueeze(2)
        hm = real.unsqueeze(1) * fmask
        avg_hm = hm.sum(dim=3, keepdim=True).sum(dim=4, keepdim=True) / (eps + fmask.sum(dim=3, keepdim=True).sum(dim=4, keepdim=True))
        
        fcolor = avg_hm * fmask
        return fcolor, fmask
    
    def _get_total_variance_loss(self, fcolor, real):
        fcolor = fcolor.sum(dim=1)
        loss = self.mse_loss(fcolor, real)
        return loss
    
    def _get_smoothness_loss(self, logits):
        if self.args.smoothness_method == 'four_l1':
            loss = 0.5 * (
                self.l1_loss(logits[:,:,:-1,:], logits[:,:,1:,:])
                + self.l1_loss(logits[:,:,:,:-1], logits[:,:,:,1:])
            )
        elif self.args.smoothness_method == 'eight_l1':
            loss = 0.25 * (
                self.l1_loss(logits[:,:,:-1,:], logits[:,:,1:,:])
                + self.l1_loss(logits[:,:,:,:-1], logits[:,:,:,1:])
                + self.l1_loss(logits[:,:,:-1,:-1], logits[:,:,1:,1:])
                + self.l1_loss(logits[:,:,1:,:-1], logits[:,:,:-1,1:])
            )
        elif self.args.smoothness_method == 'eight_l2':
            loss = 0.25 * (
                self.mse_loss(logits[:,:,:-1,:], logits[:,:,1:,:])
                + self.mse_loss(logits[:,:,:,:-1], logits[:,:,:,1:])
                + self.mse_loss(logits[:,:,:-1,:-1], logits[:,:,1:,1:])
                + self.mse_loss(logits[:,:,1:,:-1], logits[:,:,:-1,1:])
            )
        else:
            raise NotImplementedError()
        return loss

    # required
    def training_step(self, batch, batch_idx, optimizer_idx):
        base, real = get_base_real(self.args, batch)

        if self.args.tau_policy == 'decay':
            tau = self.args.min_tau ** (self.current_epoch / self.args.epochs)
        else:
            tau = self.args.min_tau

        if optimizer_idx == 0:
            # training generator
            logits, fcolor, _, _, _, pred = self(base, real, tau=tau)

            adv_g_loss = self.args.lambda_adv * self.gan_loss(self.discriminator(torch.cat((base, pred), dim=1)), True)
            rec_loss = self.args.lambda_rec * self.l1_loss(pred, real)
            tvr_loss = self.args.lambda_tvr * self._get_total_variance_loss(fcolor, real)
            smt_loss = self.args.lambda_smt * self._get_smoothness_loss(logits)
            loss = adv_g_loss + rec_loss + tvr_loss + smt_loss
            
            return {
                'loss': loss,
                'adv_g_loss': adv_g_loss,
                'rec_loss': rec_loss,
                'tvr_loss': tvr_loss,
                'smt_loss': smt_loss,
                'psnr': calculate_psnr(self.args, base, pred, real).mean()
            }
        else:
            # training discriminator
            _, _, _, _, _, pred = self(base, real, tau=tau)
            fake_loss = self.gan_loss(self.discriminator(torch.cat((base, pred), dim=1).detach()), False)
            real_loss = self.gan_loss(self.discriminator(torch.cat((base, real), dim=1)), True)
            loss = self.args.lambda_adv * 0.5 * (fake_loss + real_loss)
            return {
                'loss': loss,
                'adv_d_loss': loss
            }

    # required
    def training_epoch_end(self, outputs):
        adv_loss, rec_loss, tvr_loss, smt_loss, psnr = 0.0, 0.0, 0.0, 0.0, 0.0
        n = len(outputs[0])
        for output0, output1 in zip(outputs[0], outputs[1]):
            adv_loss += (output0['adv_g_loss'] + output1['adv_d_loss']) / n
            rec_loss += output0['rec_loss'] / n
            tvr_loss += output0['tvr_loss'] / n
            smt_loss += output0['smt_loss'] / n
            psnr += output0['psnr'] / n
        tot_loss = adv_loss + rec_loss + tvr_loss + smt_loss

        self.logger.experiment.add_scalars('train_losses', {
            'adv_loss': adv_loss,
            'rec_loss': rec_loss,
            'tvr_loss': tvr_loss,
            'smt_loss': smt_loss,
            'tot_loss': tot_loss
        }, self.current_epoch)
        self.logger.experiment.add_scalars('psnr', {'train': psnr}, self.current_epoch)

    # required
    def validation_step(self, batch, batch_idx):
        base, real = get_base_real(self.args, batch)
        _, _, _, color, mask, pred = self(base, real, tau=self.args.min_tau)

        if batch_idx == 0:
            lst = []
            for i in range(real.shape[0]):
                lst.append(get_preview_image(self.args, base[i:i+1], color[i:i+1], mask[i:i+1], pred[i:i+1], real[i:i+1]))
            
            for i, tag in enumerate(['base', 'color', 'mask', 'pred', 'real']):
                arr = np.stack([np.array(p[i]) for p in lst])/255.0
                self.logger.experiment.add_images(tag, arr, self.current_epoch, dataformats='NHWC')

        return {
            'loss': calculate_psnr(self.args, base, pred, real).mean()
        }

    # required
    def validation_epoch_end(self, outputs):
        psnr = torch.stack([output['loss'] for output in outputs]).mean()
        self.logger.experiment.add_scalars('psnr', {'val': psnr}, self.current_epoch)
        return {
            'loss': psnr,
            'checkpoint_on': psnr
        }
        
    # required
    def configure_optimizers(self):
        # optimizers
        optimizer_g = Adam(self.generator.parameters(), lr=self.args.lr)
        optimizer_d = Adam(self.discriminator.parameters(), lr=self.args.lr)
        optimizers = [
            optimizer_g,
            optimizer_d
        ]
        
        # schedulers
        def lr_lambda(epoch):
            return 1.0 - max(0, epoch - self.args.epochs * (1 - self.args.decay_ratio)) / (self.args.decay_ratio * self.args.epochs + 1)
        schedulers = [
            lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lr_lambda),
            lr_scheduler.LambdaLR(optimizer_d, lr_lambda=lr_lambda)
        ]

        return optimizers, schedulers


def get_base_real(args, batch):
    """
    get base, real in [-1.0, 1.0]
    """
    base, real = batch[0] * 2.0 - 1.0, batch[1] * 2.0 - 1.0
    return base, real


def calculate_psnr(args, base, pred, real):
    img1, img2 = (pred + 1.0) * 0.5, (real + 1.0) * 0.5
    return calculate_psnr_torch(img1, img2)


def get_preview_image(args, base, color, mask, pred, real):
    base = (base + 1.0) * 0.5
    color = ((color + 1.0) * 0.5) * mask
    pred = (pred + 1.0) * 0.5
    real = (real + 1.0) * 0.5

    # making color, mask of easy-to-print form
    color2 = color.sum(dim=1)
    mask2 = mask.sum(dim=1)

    color_map = get_color_map(mask.shape[1], mask.device)
    mask3 = mask * color_map.view(1, mask.shape[1], 3, 1, 1)
    mask3 += 1 - mask.sum(dim=1, keepdim=True)
    mask3 = mask3.sum(dim=1)

    base = base.repeat(1, 3, 1, 1)
    return rgb2image(base),\
            rgb2image(base*(1 - mask2) + color2*mask2),\
            rgb2image(mask3),\
            rgb2image(pred),\
            rgb2image(real)
