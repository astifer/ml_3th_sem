import pyrallis
from tqdm import trange
from copy import deepcopy
import numpy as np
import uuid
import os
from dataclasses import dataclass, asdict
from typing import Optional, Union, Dict, Any, Tuple
import torch
from torch.nn import functional as F
import random

import wandb
import gymnasium as gym

from src.base_modules import Actor, EnsembledCritic
from src.utils import set_seed, wandb_init, wrap_env, modify_reward, eval_actor
from src.replay_buffer import ReplayBuffer

import d4rl


@dataclass
class TrainConfig:
    project: str = "offline_rl"
    group: str = "EDAC-D4RL"
    name: str = "EDAC"
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    eta: float = 1.0
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    max_action: float = 1.0
    buffer_size: int = 1_000_000
    env_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    normalize_reward: bool = False
    eval_episodes: int = 10
    eval_every: int = 5
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    log_every: int = 100
    device: str = "cpu"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class EDAC:
    def __init__(self,
                 cfg: TrainConfig,
                 actor: Actor,
                 critic: EnsembledCritic,
                 alpha_lr: float = 3e-4) -> None:
        
        self.cfg = cfg
        self.device = cfg.device
        self.tau = cfg.tau
        self.eta = cfg.eta
        self.gamma = cfg.gamma

        self.target_entropy = -float(actor.action_dim)

        self.log_alpha = torch.tensor([0.0], dtype=torch.float32, device=self.device, requires_grad=True)

        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

        self.actor = actor.to(self.device)
        self.actor_target = deepcopy(actor).to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)

        self.critic = critic.to(self.device)
        self.critic_target = deepcopy(critic).to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)

        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, float]:
        self.total_iterations += 1

        alpha_loss = self.alpha_loss(states)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        actor_loss, batch_entropy, q_policy_std = self.actor_loss(states)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        critic_loss = self.critic_loss(states, actions, rewards, next_states, dones)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        with torch.no_grad():
            self.soft_critic_update()

            max_action = self.actor.max_action
            random_action = -max_action + 2 * max_action * torch.rand_like(actions)
            q_random_std = self.critic(states, random_action).std(0).mean().item()
        
        return {
            "alpha_loss": alpha_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "batch_entropy": batch_entropy,
            "alpha": self.alpha.item(),
            "q_policy_std": q_policy_std,
            "q_random_std": q_random_std,
        }

    def alpha_loss(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, log_prob = self.actor(state, need_log_prob=True)
        
        loss = -self.log_alpha * (log_prob + self.target_entropy)
        return loss.mean()
    
    def actor_loss(self, state: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        action, log_prob = self.actor(state, need_log_prob=True)
        q_values = self.critic(state, action)

        assert q_values.shape[0] == self.critic.num_critics

        q_value_min = q_values.min(0).values
        q_value_std = q_values.std(0).mean().item()
        batch_entropy = -log_prob.mean().item()

        assert log_prob.shape == q_value_min.shape
        loss = self.alpha * log_prob - q_value_min

        return loss.mean(), batch_entropy, q_value_std

    def critic_loss(self,
                    state: torch.Tensor,
                    action: torch.Tensor,
                    reward: torch.Tensor,
                    next_state: torch.Tensor,
                    done: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_state, need_log_prob=True)
            q_next = self.critic_target(next_state, next_action).min(0).values
            q_next = q_next - self.alpha * next_action_log_prob

            assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            q_target = reward + self.gamma * (1 - done) * q_next.unsqueeze(-1)
        
        q_values = self.critic(state, action)
        critic_loss = F.mse_loss(q_values,  q_target.squeeze(1))
        diversity_loss = self.critic_diversity_loss(state, action)

        loss = critic_loss + self.eta * diversity_loss
        return loss
    
    def critic_diversity_loss(self,
                              state: torch.Tensor,
                              action: torch.Tensor) -> torch.Tensor:
        num_critics = self.critic.num_critics

        state = state.unsqueeze(0).repeat_interleave(num_critics, dim=0)
        action = action.unsqueeze(0).repeat_interleave(num_critics, dim=0).requires_grad_(True)

        q_ensemble = self.critic(state, action)

        q_action_grad = torch.autograd.grad(q_ensemble.sum(), action, retain_graph=True, create_graph=True)[0]
        q_action_grad = q_action_grad / (torch.norm(q_action_grad, p=2, dim=2).unsqueeze(-1) + 1e-10)
        q_action_grad = q_action_grad.transpose(0, 1)

        masks = torch.eye(num_critics, device=self.device).unsqueeze(0).repeat(q_action_grad.shape[0], 1, 1)

        q_action_grad = q_action_grad @ q_action_grad.permute(0, 2, 1)
        q_action_grad = (1 - masks) * q_action_grad

        grad_loss = q_action_grad.sum(dim=(1, 2)).mean()
        grad_loss = grad_loss / (num_critics - 1)

        return grad_loss
    
    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "actor_optimizer": self.actor_optim.state_dict(),
            "critic_optimizer": self.critic_optim.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["target_critic"])
        self.actor_optim.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optim.load_state_dict(state_dict["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        self.log_alpha.data[0] = state_dict["log_alpha"]
        self.alpha = self.log_alpha.exp().detach()


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    wandb_init(asdict(config))

    # data, evaluation, env setup
    eval_env = wrap_env(gym.make(config.env_name))
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    d4rl_dataset = d4rl.qlearning_dataset(eval_env)

    if config.normalize_reward:
        modify_reward(d4rl_dataset, config.env_name)

    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=config.device,
    )
    buffer.load_d4rl_dataset(d4rl_dataset)

    # Actor & Critic setup
    actor = Actor(state_dim, action_dim, config.hidden_dim, config.max_action)
    actor.to(config.device)
    critic = EnsembledCritic(
        state_dim, action_dim, config.hidden_dim, config.num_critics
    )
    critic.to(config.device)

    trainer = EDAC(
        cfg=config,
        actor=actor,
        critic=critic,
        alpha_learning_rate=config.alpha_learning_rate,
    )
    # saving config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    total_updates = 0.0
    for epoch in trange(config.num_epochs, desc="Training"):
        # training
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = buffer.sample(config.batch_size)
            update_info = trainer.update(batch)

            if total_updates % config.log_every == 0:
                wandb.log({"epoch": epoch, **update_info})

            total_updates += 1

        # evaluation
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = eval_actor(
                env=eval_env,
                actor=actor,
                n_episodes=config.eval_episodes,
                seed=config.eval_seed,
                device=config.device,
            )
            eval_log = {
                "eval/reward_mean": np.mean(eval_returns),
                "eval/reward_std": np.std(eval_returns),
                "epoch": epoch,
            }
            if hasattr(eval_env, "get_normalized_score"):
                normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
                eval_log["eval/normalized_score_mean"] = np.mean(normalized_score)
                eval_log["eval/normalized_score_std"] = np.std(normalized_score)

            wandb.log(eval_log)

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"{epoch}.pt"),
                )

    wandb.finish()


if __name__ == "__main__":
    train()
