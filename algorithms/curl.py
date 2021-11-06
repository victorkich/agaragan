from models import ActorCURL, CriticCURL, CURL
from utils.utils import soft_update_params
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


class CurlSAC(object):
    """CURL representation learning with SAC."""
    def __init__(self, obs_shape, action_shape, config, save_dir, writer, algorithm='SAC'):
        self.writer = writer
        self.save_dir = save_dir
        self.device = config['device']
        self.discount = config['discount_rate']
        self.critic_tau = config['tau']
        self.encoder_tau = config['tau']
        self.actor_update_freq = config['actor_update_freq']
        self.critic_target_update_freq = config['critic_target_update_freq']
        self.cpc_update_freq = config['cpc_update_freq']
        self.curl_latent_dim = config['curl_latent_dim']
        self.detach_encoder = config['detach_encoder']

        self.actor = ActorCURL(obs_shape, action_shape, config['hidden_dim'], 'pixel', config['encoder_feature_dim'],
                               config['actor_log_std_min'], config['actor_log_std_max'], config['num_layers'],
                               config['num_filters']).to(config['device'])

        self.critic = CriticCURL(obs_shape, action_shape, config['hidden_dim'], 'pixel', config['encoder_feature_dim'],
                                 config['num_layers'], config['num_filters']).to(config['device'])

        self.critic_target = CriticCURL(obs_shape, action_shape, config['hidden_dim'], 'pixel',
                                        config['encoder_feature_dim'], config['num_layers'],
                                        config['num_filters']).to(config['device'])

        self.critic_target.load_state_dict(self.critic.state_dict())

        # Tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(config['init_temperature'])).to(config['device'])
        self.log_alpha.requires_grad = True
        # Set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_learning_rate'],
                                                betas=(config['actor_beta'], 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_learning_rate'],
                                                 betas=(config['critic_beta'], 0.999))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config['alpha_learning_rate'],
                                                    betas=(config['alpha_beta'], 0.999))

        # Create CURL encoder (the 128 batch size is probably unnecessary)
        self.CURL = CURL(obs_shape, config['encoder_feature_dim'], self.curl_latent_dim, self.critic,
                         self.critic_target, output_type='continuous').to(self.device)

        # Optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(self.critic.encoder.parameters(), lr=config['encoder_learning_rate'])
        self.cpc_optimizer = torch.optim.Adam(self.CURL.parameters(), lr=config['encoder_learning_rate'])
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.Tensor = torch.cuda.FloatTensor if config['device'] == 'cuda' else torch.Tensor

        self.train()
        self.critic_target.train()
        self.num_training = 0

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.CURL.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = obs.type(self.Tensor)
            # obs = torch.FloatTensor(obs).to(self.device)
            # obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_log_pi=False)
            return mu.cpu().data.numpy().flatten()

    def forward(self, obs):
        obs = obs.type(self.Tensor)
        # obs = torch.FloatTensor(obs).to(self.device)
        mu, _, _, _ = self.actor(obs, compute_log_pi=False)
        return mu

    def sample_action(self, obs):
        with torch.no_grad():
            obs = obs.type(self.Tensor)
            # obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Log metrics
        self.writer.add_scalar(tag="Critic Loss", scalar_value=critic_loss.item(), global_step=self.num_training)
        return critic_loss.cpu().detach().numpy()

    def update_actor_and_alpha(self, obs):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
        # entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # Log metrics
        self.writer.add_scalar(tag="Actor Loss", scalar_value=actor_loss.item(), global_step=self.num_training)

    def update_cpc(self, obs_anchor, obs_pos):
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)

        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()

        # Log metrics
        self.writer.add_scalar(tag="Encoder Loss", scalar_value=loss.item(), global_step=self.num_training)

    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_cpc()
        critic_loss = self.update_critic(obs, action, reward, next_obs, not_done)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
            soft_update_params(self.critic.encoder, self.critic_target.encoder, self.encoder_tau)
        
        if step % self.cpc_update_freq == 0:
            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            self.update_cpc(obs_anchor, obs_pos)

        self.num_training += 1
        return critic_loss

    def save(self):
        torch.save(self.CURL.state_dict(), self.save_dir + 'curl_%d.pth' % self.num_training)

    def load(self):
        self.CURL.load_state_dict(torch.load(self.save_dir + 'curl.pth'))
