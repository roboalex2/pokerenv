"""Train a No Limit Hold'em self-play bot in pokerenv.

The training loop keeps a pool of older policy snapshots and samples opponents from
that pool. The latest policy (hero) is optimized with a simple actor-critic loss.
If CUDA is available (for example on an RTX 3070) it is used automatically.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

import pokerenv.obs_indices as indices
from pokerenv.common import Action, PlayerAction
from pokerenv.table import Table


OBS_SIZE = 58
N_ACTIONS = 4


class PolicyValueNet(nn.Module):
    def __init__(self, obs_size: int = OBS_SIZE, hidden_size: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_size, N_ACTIONS)
        self.bet_mean_head = nn.Linear(hidden_size, 1)
        self.bet_log_std = nn.Parameter(torch.zeros(1))
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        action_logits = self.action_head(x)
        bet_mean = self.bet_mean_head(x)
        value = self.value_head(x)
        return action_logits, bet_mean, self.bet_log_std.expand_as(bet_mean), value


@dataclass
class StepSample:
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


class LearningAgent:
    def __init__(self, model: PolicyValueNet, optimizer: torch.optim.Optimizer, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.samples: List[StepSample] = []

    def reset_hand(self):
        self.samples.clear()

    def act(self, obs: np.ndarray) -> Action:
        action, sample = sample_action(self.model, obs, self.device, training=True)
        if sample is not None:
            self.samples.append(sample)
        return action

    def finish_hand(self, final_reward: float, entropy_coef: float, value_coef: float, grad_clip: float):
        if not self.samples:
            return {"loss": 0.0, "reward": float(final_reward)}

        reward_t = torch.tensor([final_reward], dtype=torch.float32, device=self.device)
        policy_losses = []
        value_losses = []
        entropies = []

        for sample in self.samples:
            advantage = reward_t - sample.value.squeeze(-1)
            policy_losses.append(-sample.log_prob * advantage.detach())
            value_losses.append(advantage.pow(2))
            entropies.append(sample.entropy)

        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy_bonus = torch.stack(entropies).mean()
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.optimizer.step()

        return {
            "loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_bonus.item()),
            "reward": float(final_reward),
        }


class FrozenPolicyAgent:
    def __init__(self, model: PolicyValueNet, device: torch.device):
        self.model = model
        self.device = device

    def act(self, obs: np.ndarray) -> Action:
        with torch.no_grad():
            action, _ = sample_action(self.model, obs, self.device, training=False)
        return action


class RandomAgent:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def act(self, obs: np.ndarray) -> Action:
        valid_actions = np.argwhere(obs[indices.VALID_ACTIONS] == 1).flatten()
        chosen = PlayerAction(int(self.rng.choice(valid_actions)))
        if chosen is PlayerAction.BET:
            low = float(obs[indices.VALID_BET_LOW])
            high = float(obs[indices.VALID_BET_HIGH])
            bet = float(self.rng.uniform(low, high)) if high > low else low
        else:
            bet = 0.0
        return Action(chosen, bet)


def sample_action(model: PolicyValueNet, obs: np.ndarray, device: torch.device, training: bool):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits, bet_mean, bet_log_std, value = model(obs_t)

    valid_mask = torch.tensor(obs[indices.VALID_ACTIONS], dtype=torch.float32, device=device).unsqueeze(0)
    masked_logits = logits.masked_fill(valid_mask == 0, -1e9)
    action_dist = Categorical(logits=masked_logits)
    action_idx = action_dist.sample()

    bet_low = float(obs[indices.VALID_BET_LOW])
    bet_high = float(obs[indices.VALID_BET_HIGH])
    chosen_action = PlayerAction(int(action_idx.item()))
    bet_amount = 0.0

    total_log_prob = action_dist.log_prob(action_idx)
    entropy = action_dist.entropy()

    if chosen_action is PlayerAction.BET and bet_high > bet_low:
        bet_std = bet_log_std.exp().clamp_min(1e-3)
        bet_dist = Normal(loc=bet_mean, scale=bet_std)
        raw_bet = bet_dist.rsample()
        bet_unit = torch.sigmoid(raw_bet)
        bet = bet_low + (bet_high - bet_low) * bet_unit
        bet_amount = float(bet.item())
        total_log_prob = total_log_prob + bet_dist.log_prob(raw_bet).sum(-1)
        entropy = entropy + bet_dist.entropy().sum(-1)
    elif chosen_action is PlayerAction.BET:
        bet_amount = bet_low

    sample = None
    if training:
        sample = StepSample(log_prob=total_log_prob.squeeze(0), value=value.squeeze(0), entropy=entropy.squeeze(0))

    return Action(chosen_action, bet_amount), sample


def build_frozen_agent_from_state(
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
    hidden_size: int,
) -> FrozenPolicyAgent:
    model = PolicyValueNet(hidden_size=hidden_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return FrozenPolicyAgent(model, device)


def train(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (CUDA not available or --cpu was set).")

    table = Table(
        n_players=args.players,
        stack_low=args.stack_low,
        stack_high=args.stack_high,
        invalid_action_penalty=args.invalid_action_penalty,
    )
    table.seed(args.seed)

    hero_model = PolicyValueNet(hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(hero_model.parameters(), lr=args.lr)
    hero_agent = LearningAgent(hero_model, optimizer, device)

    snapshot_pool: List[Optional[Dict[str, torch.Tensor]]] = [None]
    recent_rewards: List[float] = []

    for hand_idx in range(1, args.hands + 1):
        obs = table.reset()
        hero_agent.reset_hand()

        opponent_agents: Dict[int, object] = {}
        for pid in range(1, args.players):
            snapshot = snapshot_pool[int(rng.integers(0, len(snapshot_pool)))]
            if snapshot is None:
                opponent_agents[pid] = RandomAgent(rng)
            else:
                opponent_agents[pid] = build_frozen_agent_from_state(snapshot, device, args.hidden_size)

        done = False
        final_reward = 0.0
        while not done:
            acting_player = int(obs[indices.ACTING_PLAYER])
            if acting_player == 0:
                action = hero_agent.act(obs)
            else:
                action = opponent_agents[acting_player].act(obs)

            obs, reward, done, _ = table.step(action)
            if done:
                final_reward = float(reward[0])

        metrics = hero_agent.finish_hand(
            final_reward=final_reward,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            grad_clip=args.grad_clip,
        )
        recent_rewards.append(final_reward)
        if len(recent_rewards) > args.log_window:
            recent_rewards.pop(0)

        if hand_idx % args.snapshot_interval == 0:
            state = {k: v.detach().cpu().clone() for k, v in hero_model.state_dict().items()}
            snapshot_pool.append(state)
            if len(snapshot_pool) > args.max_snapshots + 1:
                snapshot_pool.pop(1)

        if hand_idx % args.log_every == 0:
            avg_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            print(
                f"hand={hand_idx:6d} avg_reward={avg_reward:+.4f} "
                f"loss={metrics['loss']:+.4f} pool={len(snapshot_pool)-1}"
            )

    torch.save(hero_model.state_dict(), args.output)
    print(f"Saved trained model to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Self-play trainer for no-limit hold'em using pokerenv")
    parser.add_argument("--hands", type=int, default=50000, help="Number of training hands")
    parser.add_argument("--players", type=int, default=6, choices=[2, 3, 4, 5, 6], help="Players at the table")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden layer width")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--snapshot-interval", type=int, default=500, help="Hands between opponent snapshots")
    parser.add_argument("--max-snapshots", type=int, default=20, help="Maximum historical snapshots to keep")
    parser.add_argument("--stack-low", type=int, default=50, help="Minimum stack in big blinds")
    parser.add_argument("--stack-high", type=int, default=200, help="Maximum stack in big blinds")
    parser.add_argument("--invalid-action-penalty", type=float, default=0.01, help="Penalty for invalid actions")
    parser.add_argument("--log-every", type=int, default=100, help="How often to print training logs")
    parser.add_argument("--log-window", type=int, default=500, help="Window size for avg reward logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="self_play_policy.pt", help="Path for the trained model")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
