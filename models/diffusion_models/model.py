import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import models.diffusion_models.networks as networks
from models.base_model import BaseModel

logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        """Initialize the DDPM model settings and pretrained networks.

        Args:
            opt: A dictionary containing all the configuration information.
        """
        super(DDPM, self).__init__(opt)
        self.netG = self.set_device(networks.define_diffusion(opt))
        self.schedule_phase = None

        self.months = []  # A list of months of curent data given by feed_data.

        self.set_loss()  # Initialize loss functions.
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            if opt['model']['finetune_norm']:
                optim_params = []  # Parameters for optimization if finetuning normalization layers.
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info('Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())
            # optimizer cant be set
            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()  # Dictionary to store logs for visualization and monitoring.
        self.load_network()  # Load network if a checkpoint is provided.
        self.print_network()  # Print network details.

    def feed_data(self, data: tuple) -> None:
        """Stores data for feeding into the model and month indices for each tensor in a batch.

        Args:
            data: A tuple containing dictionary with the following keys: (
                {HR: a batch of high-resolution images [B, C, H, W]},
                {LR: a batch of low-resolution images [B, C, H, W]},
                {INTERPOLATED: a batch of upsampled (via interpolation) images [B, C, H, W]},
                [list of corresponding months of samples in a batch]
        """
        self.data, self.months = self.set_device(data[0]), data[1]

    def optimize_parameters(self):
        """Perform a single optimization step for the model."""
        self.optG.zero_grad()  # Zero gradients.
        l_pix = self.netG(self.data)  # Calculate loss.
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum() / int(b * c * h * w)  # Average the loss over all pixels.
        l_pix.backward()  # Backpropagate.
        self.optG.step()  # Update weights.
        self.log_dict['l_pix'] = l_pix.item()  # Store loss in log dictionary.

    def generate_sr(self, continous=False):
        """Generate super-resolution images using the model.

        Args:
            continous: Whether to save ouput for each iteration T while sampling.
        """
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(self.data, continous)
            else:
                self.SR = self.netG.super_resolution(self.data, continous)
            self.SR = self.SR.unsqueeze(0) if len(self.SR.size()) == 3 else self.SR

        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        """Generate samples from the model.

        Args:
            batch_size: Number of samples to generate.
            continous: Whether to save ouput for each iteration T while sampling.
        """
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        """Initialize the loss function for the model."""
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        """Set the noise schedule for training or evaluation.

        Args:
            schedule_opt: Options for the noise schedule ('quad', 'linear', 'warmup10', 'const', 'jsd', cosine).
            schedule_phase: Indicates if it's for 'train' or 'val' phase.
        """
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        """Retrieve the current log dictionary containing loss values and other metrics.

        Returns:
            OrderedDict: A dictionary containing the current log values.
        """
        return self.log_dict

    def get_images(self, need_LR=True, sample=False):
        """Get the current visuals for monitoring and validation.

        Args:
            need_LR: Boolean indicating whether low-resolution images are needed.
            sample: Boolean indicating whether to return samples instead of results.

        Returns:
            OrderedDict: A dictionary containing the current visuals (images).
        """
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        """Print the detailed network structure and parameters to the log."""
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        """Save the network and optimizer states to disk.

        Args:
            epoch: Current epoch of training.
            iter_step: Current iteration step within the epoch.
        """
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        """Load network and optimizer states from disk if a checkpoint is provided."""

        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

    def prepare_to_train(self) -> None:
        """Sets model to train mode and activates implementation-specific training procedures.
        """
        self.set_new_noise_schedule(self.opt['model']['beta_schedule']['train'], schedule_phase='train')

    def prepare_to_eval(self) -> None:
        """Sets model to eval mode and activates implementation-specific evaluation procedures.
        """
        self.set_new_noise_schedule(self.opt['model']['beta_schedule']['val'], schedule_phase='val')

    def get_months(self) -> list:
        """Returns the list of month indices corresponding to batch of samples
        fed into the model with feed_data.

        Returns:
            months: Current list of months.
        """
        return self.months

    def get_loaded_iter(self) -> int | None:
        """Get the loaded iteration number from a saved state.

        Returns:
            int | None: The iteration number if available, otherwise None.
        """
        return self.begin_step

    def get_loaded_epoch(self) -> int | None:
        """Get the loaded epoch number from a saved state.

        Returns:
            int | None: The epoch number if available, otherwise None.
        """
        return self.begin_epoch
