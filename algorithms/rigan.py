from utils.utils import ReplayBufferGAN, Variable, LambdaLR, weights_init_normal, DatasetGAN
from torchvision.utils import save_image, make_grid
from models import Discriminator, GeneratorResNet
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_s
from torch.optim import Adam
import numpy as np
import itertools
import torch
import sys


class RiGAN(object):
    def __init__(self, input_shape, config, save_dir, writer):
        self.writer = writer
        self.save_dir = save_dir
        self.input_shape = input_shape
        n_residual_blocks = config['n_residual_blocks']  # number of residual blocks in generator
        lr = config['rigan_learning_rate']  # learning rate
        b1 = config['b1']  # decay of first order momentum of gradient
        b2 = config['b2']  # decay of first order momentum of gradient
        n_workers = config['n_workers']  # number of cpu threads to use during batch generation
        self.lambda_cyc = config['lambda_cyc']  # cycle loss weight
        self.lambda_id = config['lambda_id']  # identity loss weight
        self.lambda_rl = config['lambda_rl']  # rl loss weight
        self.lambda_scene = config['lambda_scene']  # rl scene loss weight
        self.lambda_gan = config['lambda_gan']  # gan loss weight
        self.checkpoint_interval = config['num_episode_save']  # interval between saving model checkpoints
        self.sample_interval = config['sample_interval']  # interval between saving generator outputs
        self.epoch = 0  # counter for number of epochs
        n_epochs = config['num_eps_train']  # number of epochs of training
        decay_epoch = config['decay_epoch']  # epoch from which to start lr decay
        batch_size = 1  # size of the batches

        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_scene = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        # Initialize generator and discriminator
        self.G_AB = GeneratorResNet(input_shape, n_residual_blocks)
        self.G_BA = GeneratorResNet(input_shape, n_residual_blocks)
        self.D_A = Discriminator(input_shape)
        self.D_B = Discriminator(input_shape)

        if config['device'] == 'cuda':
            self.G_AB = self.G_AB.cuda()
            self.G_BA = self.G_BA.cuda()
            self.D_A = self.D_A.cuda()
            self.D_B = self.D_B.cuda()
            self.criterion_GAN.cuda()
            self.criterion_cycle.cuda()
            self.criterion_identity.cuda()
            self.criterion_scene.cuda()

        # Initialize weights
        self.G_AB.apply(weights_init_normal)
        self.G_BA.apply(weights_init_normal)
        self.D_A.apply(weights_init_normal)
        self.D_B.apply(weights_init_normal)

        # Optimizers
        self.optimizer_G = Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=lr, betas=(b1, b2))
        self.optimizer_D_A = Adam(self.D_A.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_D_B = Adam(self.D_B.parameters(), lr=lr, betas=(b1, b2))

        # Learning rate update schedulers
        self.lr_scheduler_G = lr_s.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(n_epochs, self.epoch, decay_epoch).step)
        self.lr_scheduler_D_A = lr_s.LambdaLR(self.optimizer_D_A, lr_lambda=LambdaLR(n_epochs, self.epoch, decay_epoch).step)
        self.lr_scheduler_D_B = lr_s.LambdaLR(self.optimizer_D_B, lr_lambda=LambdaLR(n_epochs, self.epoch, decay_epoch).step)

        self.Tensor = torch.cuda.FloatTensor if config['device'] == 'cuda' else torch.Tensor

        # Buffers of previously generated samples
        self.fake_A_buffer = ReplayBufferGAN()
        self.fake_B_buffer = ReplayBufferGAN()

        # Image transformations
        transform = [
            transforms.Resize((input_shape[1], input_shape[2])),
            transforms.RandomCrop((input_shape[1], input_shape[2])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.transforms = transforms.Compose(transform)
        # Image transformations
        transform2 = [
            transforms.Resize((input_shape[1], input_shape[2])),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.transforms2 = transforms.Compose(transform2)

        # Loading the train and val dataset using data loader
        dataset = DatasetGAN(transform=self.transforms, dir=save_dir)
        lengths = [round(len(dataset) * 0.8), round(len(dataset) * 0.2)]
        train_data = Subset(dataset, range(0, lengths[0]))
        val_data = Subset(dataset, range(lengths[0], sum(lengths)))
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=1)

    def forward(self, obs):
        image = obs.type(self.Tensor)
        return self.G_AB(image)

    def update(self, agent, loss_rl, obs):
        batch = next(iter(self.train_loader))
        # Set model input
        real_A = Variable(batch.type(self.Tensor))
        real_B = Variable(obs.type(self.Tensor))
        # Adversarial ground truths
        valid = Variable(self.Tensor(np.ones((real_A.size(0), *self.D_A.output_shape))), requires_grad=False)
        fake = Variable(self.Tensor(np.zeros((real_A.size(0), *self.D_A.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------
        self.G_AB.train()
        self.G_BA.train()
        self.optimizer_G.zero_grad()

        # Identity loss
        # loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
        # loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)
        # loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = self.G_AB(real_A)
        loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
        fake_A = self.G_BA(real_B)
        loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = self.G_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(recov_A, real_A)
        recov_B = self.G_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Scene loss
        fake_B1 = self.G_BA(real_B)
        fake_B2 = self.G_AB(fake_B1)
        loss_scene = self.criterion_scene(agent.forward(real_B), agent.forward(fake_B1)) + \
                     self.criterion_scene(agent.forward(real_B), agent.forward(fake_B2)) + \
                     self.criterion_scene(agent.forward(fake_B1), agent.forward(fake_B2))

        # Total loss
        # loss_G = loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity
        loss_G = self.lambda_gan * loss_GAN + self.lambda_cyc * loss_cycle + \
                 self.lambda_scene * loss_scene + self.lambda_rl * loss_rl
        loss_G.backward()
        self.optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------
        self.optimizer_D_A.zero_grad()
        # Real loss
        loss_real = self.criterion_GAN(self.D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
        loss_fake = self.criterion_GAN(self.D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        self.optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        self.optimizer_D_B.zero_grad()

        # Real loss
        loss_real = self.criterion_GAN(self.D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
        loss_fake = self.criterion_GAN(self.D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        self.optimizer_D_B.step()
        loss_D = (loss_D_A + loss_D_B) / 2

        # Print log
        sys.stdout.write("\r[D loss: %f] [G loss: %f, adv: %f, cycle: %f, rl: %f, rl-scene: %f]" % (loss_D.item(),
                         loss_G.item(), loss_GAN.item(), loss_cycle.item(), loss_rl.item(), loss_scene.item()))
        self.writer.add_scalars(main_tag="Loss", tag_scalar_dict={"Discriminator": loss_D.item(),
                                                                  "Generator": loss_G.item(), "GAN": loss_GAN.item(),
                                                                  "Cycle": loss_cycle.item(), "RL": loss_rl.item(),
                                                                  "Scene": loss_scene.item()}, global_step=self.epoch)

        # If at sample interval save image
        if self.epoch % self.sample_interval == 0:
            self.sample_images(obs)

        # Update learning rates
        self.lr_scheduler_G.step()
        self.lr_scheduler_D_A.step()
        self.lr_scheduler_D_B.step()

        self.epoch += 1

    def save(self):
        # Save model checkpoints
        torch.save(self.G_AB.state_dict(), self.save_dir + 'G_AB_%d.pth' % self.epoch)
        torch.save(self.G_BA.state_dict(), self.save_dir + 'G_BA_%d.pth' % self.epoch)
        torch.save(self.D_A.state_dict(), self.save_dir + 'D_A_%d.pth' % self.epoch)
        torch.save(self.D_B.state_dict(), self.save_dir + 'D_B_%d.pth' % self.epoch)

    def load(self):
        # Load model checkpoints
        self.G_AB.load_state_dict(self.save_dir + 'G_AB.pth')
        self.G_BA.load_state_dict(self.save_dir + 'G_BA.pth')
        self.D_A.load_state_dict(self.save_dir + 'D_A.pth')
        self.D_B.load_state_dict(self.save_dir + 'D_B.pth')

    def sample_images(self, obs):
        # Saves a generated sample from the test set
        imgs = next(iter(self.val_loader))
        self.G_AB.eval()
        self.G_BA.eval()
        real_A = Variable(imgs.type(self.Tensor))
        fake_B = self.transforms2(self.G_AB(real_A))
        real_B = obs.type(self.Tensor)
        fake_A = self.transforms2(self.G_BA(real_B))

        # Arange images along x-axis
        image_grid = make_grid([real_A, real_B, fake_A, fake_B], nrow=4, normalize=True)
        # real_B = make_grid(real_B, nrow=1, normalize=True)
        # fake_A = make_grid(fake_A, nrow=1, normalize=True)
        # fake_B = make_grid(fake_B, nrow=1, normalize=True)

        # Arange images along y-axis
        # image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        # save_image(image_grid, "outputs/%s.png" % self.epoch, normalize=False)

        # Log metrics
        self.writer.add_image(tag="Image", img_tensor=image_grid, global_step=self.epoch)
