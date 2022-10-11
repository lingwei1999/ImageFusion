import logging
from functools import reduce
from pathlib import Path

import numpy as np

import torch.nn.functional as F
import torch

from kornia.losses import SSIMLoss
from kornia.metrics import AverageMeter
from torch import nn, Tensor
from torch.optim import RMSprop, Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from functions.div_loss import div_loss
# from models.Unet import UNet
from models.net_densefuse import DenseFuse_net

from utils.environment_probe import EnvironmentProbe
from utils.fusion_data import FusionData


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


class Train:
    """
    The train process for TarDAL.
    """

    def __init__(self, environment_probe: EnvironmentProbe, config: dict):
        logging.info(f'Training')
        logging.info(f'ID: {config.id}')
        self.config = config
        self.environment_probe = environment_probe

        # modules
        # self.net = UNet()
        self.net = DenseFuse_net()
        para = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        logging.info('Model params: {:}'.format(para))

        # WGAN adam optim
        logging.info(f'RMSprop | learning rate: {config.learning_rate}')
        self.opt_net = RMSprop(self.net.parameters(), lr=config.learning_rate)



        # move to device
        logging.info(f'module device: {environment_probe.device}')

        self.net.to(environment_probe.device)

        # loss
        
        self.L1Loss = torch.nn.L1Loss()
        self.L1Loss.cuda()
        self.sobelconv = Sobelxy()

        # datasets
        folder = Path(config.folder)
        resize = transforms.Resize((config.size, config.size))
        train_dataset = FusionData(folder, mode='train', transforms=resize)
        self.train_dataloader = DataLoader(train_dataset, config.batch_size, True, num_workers=config.num_workers, pin_memory=True)

        eval_dataset = FusionData(folder, mode='val', transforms=resize)
        self.eval_dataloader = DataLoader(eval_dataset, config.batch_size, False, num_workers=config.num_workers, pin_memory=True)

        logging.info(f'dataset | folder: {str(folder)} | train size: {len(self.train_dataloader) * config.batch_size}')
        logging.info(f'dataset | folder: {str(folder)} |   val size: {len(self.eval_dataloader) * config.batch_size}')

    def int_loss(self, inputs, fusion):
        assert 0



    def train_discriminator(self, input: Tensor) -> Tensor:
        """
        Train discriminator
        """

        logging.debug('train discriminator')
        # switch to train mode
        self.discriminator.train()

        # sample real & fake
        real_s = input
        self.net_ext.eval()
        self.net_con.eval()
        self.net_att.eval()

        input_1, input_b_1, input_b_2 = self.net_ext(input)
        input_att = self.net_att(input)
        fus_1 = input_1 * input_att
        fake_s = self.net_con(fus_1, input_b_1, input_b_2)

        # judge value towards real & fake
        real_v = torch.squeeze(self.discriminator(real_s))
        fake_v = torch.squeeze(self.discriminator(fake_s))

        # loss calculate
        real_l, fake_l = -real_v.mean(), fake_v.mean()

        loss = real_l + fake_l


        # backwardD
        self.opt_discriminator.zero_grad()
        loss.backward()
        self.opt_discriminator.step()

        return loss.item()

    def train_generator(self, ir, vi) -> dict:
        """
        Train generator 'ir + vi -> fus'
        """

        logging.debug('train generator')
        
        self.net.train()


        fusion = self.net(ir, vi)

        # calculate loss towards criterion
        l_int = self.L1Loss(fusion, torch.max(ir, vi))
        
        vi_grad=self.sobelconv(vi)
        ir_grad=self.sobelconv(ir)
        fusion_grad=self.sobelconv(fusion)

        l_grad = self.L1Loss(fusion_grad, torch.max(ir_grad, vi_grad))
        loss = l_int + 10*l_grad

        # backwardG
        self.opt_net.zero_grad()
        loss.backward()
        self.opt_net.step()

        state = {
            'g_loss': loss.item(),
            'g_l_int': l_int.item(),
            'g_l_grad': l_grad.item()*10,
        }
        return state

    def eval_generator(self, ir, vi) -> dict:
        """
        Eval generator
        """

        logging.debug('eval generator')
        
        self.net.eval()

        fusion = self.net(ir, vi)

        l_int = self.L1Loss(fusion, torch.max(ir, vi))
        
        vi_grad=self.sobelconv(vi)
        ir_grad=self.sobelconv(ir)
        fusion_grad=self.sobelconv(fusion)

        l_grad = self.L1Loss(fusion_grad, torch.max(ir_grad, vi_grad))
        loss = l_int + 10*l_grad

        return loss.item()

    def run(self):   
        best = float('Inf')
        best_epoch = 0

        for epoch in range(1, self.config.epochs + 1):
            train_process = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
            
            meter = AverageMeter()
            for idx, train_sample in train_process:
                
                train_ir = train_sample['ir']
                train_vis = train_sample['vi']

                train_ir = train_ir.to(self.environment_probe.device)
                train_vis = train_vis.to(self.environment_probe.device)

                g_loss = self.train_generator(train_ir, train_vis)
                # d_loss = self.train_discriminator(train_im)

                meter.update(Tensor(list(g_loss.values())))
                train_process.set_description(f'g: {g_loss["g_loss"]:03f} | l_int: {g_loss["g_l_int"]:03f} | l_grad: {g_loss["g_l_grad"]:03f}')

            # Eval Generator
            total_loss = 0
            eval_process = tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader))
            for idx, eval_sample in eval_process:
                eval_ir = eval_sample['ir']
                eval_vi = eval_sample['vi']
                eval_ir = eval_ir.to(self.environment_probe.device)
                eval_vi = eval_vi.to(self.environment_probe.device)

                loss = self.eval_generator(eval_ir, eval_vi)
                total_loss += loss

            
            logging.info(f'[{epoch}] g_loss: {meter.avg[0]:03f}')

            if epoch % 1 == 0:

                if best > meter.avg[0]:
                    best = meter.avg[0]
                    best_epoch = epoch

                    self.save(epoch, is_best=True)

                self.save(epoch)
                logging.info(f'best epoch is {best_epoch}, loss is {best}')



    def save(self, epoch: int, is_best = False):
        path = Path(self.config.cache) / self.config.id
        path.mkdir(parents=True, exist_ok=True)

        if is_best:
            cache = path / f'best.pth'
        else:
            cache = path / f'{epoch:03d}.pth'
        logging.info(f'save checkpoint to {str(cache)}')

        state = {
            'net': self.net.state_dict(),
        }

        torch.save(state, cache)
