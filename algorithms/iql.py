import pyrallis
from dataclasses import dataclass, asdict
import uuid
from typing import Optional, Dict
import os
from copy import deepcopy
import torch
from torch.nn import functional as F

import wandb
import gym
import d4rl

from src.base_modules import EnsembledCritic, BerkleyActor, ValueFunction
from src.utils import set_seed, modify_reward, compute_mean_std, wrap_env, normalize_states, wandb_init, eval_actor
from src.replay_buffer import ReplayBuffer


@dataclass
class TrainConfig:
    project: str = "offline_rl"
    group: str = "IQL-D4RL"
    name: str = "IQL"
    env: str = "halfcheetah-medium-expert-v2"
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 3.0
    iql_tau: float = 0.7
    iql_deterministic: bool = False
    max_timesteps: int = int(1e6)
    buffer_size: int = 2_000_000
    batch_size: int = 256
    normalize: bool = True
    normalize_reward: bool = False
    value_func_lr: float = 3e-4
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    actor_dropout: Optional[float] = None
    eval_freq: int = int(5e3)
    n_episodes: int = 10
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    seed: int = 0
    device: str = "cuda"

    state_dim: int = 17
    action_dim: int = 6
    hidden_dim: int = 256
    exp_adv_max: float = 100.0

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class IQL:
    def __init__(self,
                 cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.actor = BerkleyActor(cfg.state_dim,
                           cfg.action_dim,
                           cfg.hidden_dim).to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optim, cfg.max_timesteps)

        self.critic = EnsembledCritic(cfg.state_dim,
                                      cfg.action_dim,
                                      cfg.hidden_dim).to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)
        
        with torch.no_grad():
            self.critic_target = deepcopy(self.critic).to(self.device)
        
        self.value_func = ValueFunction(cfg.state_dim,
                                        cfg.hidden_dim).to(self.device)
        self.value_optim = torch.optim.AdamW(self.value_func.parameters(), lr=cfg.value_func_lr)

        self.iql_tau = cfg.iql_tau
        self.tau = cfg.tau
        self.discount = cfg.discount
        self.beta = cfg.beta
        self.exp_adv_max = cfg.exp_adv_max

        self.total_iterations = 0
    
    def assymetric_l2(self, u: torch.Tensor) -> torch.Tensor:
        loss = torch.abs(self.iql_tau - (u < 0).float()) * u.pow(2)

        return loss.mean()
    
    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * tgt_param.data)
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, float]:
        
        self.total_iterations += 1

        with torch.no_grad():
            next_v = self.value_func(next_states)
            v_target = self.critic_target(states, actions).min(0).values

            q_target = rewards + (1.0 - dones) * self.discount * next_v
        
        # value func step
        value = self.value_func(states)
        advantage = v_target - value
        value_loss = self.assymetric_l2(advantage)

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        # critic step
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, q_target.squeeze(-1))

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor step
        exp_advantage = torch.exp(self.beta * advantage.detach()).clamp_max(self.exp_adv_max)

        bc_losses = -self.actor.log_prob(states, actions)
        actor_loss = (bc_losses * exp_advantage).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.lr_scheduler.step()

        return {
            "value_loss": value_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "actor_learning_rate": self.lr_scheduler.get_last_lr()[0]
        }


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = IQL(config)

    # if config.load_model != "":
    #     policy_file = Path(config.load_model)
    #     trainer.load_state_dict(torch.load(policy_file))
    #     actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        states, actions, rewards, next_states, dones = replay_buffer.sample(config.batch_size)
        # batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(states, actions, rewards, next_states, dones)
        wandb.log(log_dict, step=trainer.total_iterations)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                trainer.actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score}, step=trainer.total_iterations
            )


if __name__ == "__main__":
    train()
