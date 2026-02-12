"""Train a No Limit Hold'em bot with a Deep CFR-style loop.

This implementation adapts Deep CFR concepts to pokerenv:
- finite action abstraction with discrete bet buckets,
- external-sampling traversals for regret samples,
- advantage networks trained on reservoir buffers,
- average strategy network trained from sampled behavior policies.

It auto-uses CUDA when available (e.g. RTX 3070).
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pokerenv.obs_indices as indices
from pokerenv.common import Action, PlayerAction
from pokerenv.table import Table


OBS_SIZE = 58


@dataclass(frozen=True)
class DiscreteAction:
    kind: str
    bucket_i: int = -1


class ActionAbstraction:
    """Maps pokerenv actions to a finite discrete action set for CFR."""

    def __init__(self, bet_fractions: List[float]):
        self.bet_fractions = bet_fractions
        self.actions: List[DiscreteAction] = [
            DiscreteAction("FOLD"),
            DiscreteAction("CHECK"),
            DiscreteAction("CALL"),
            *[DiscreteAction("BET", i) for i in range(len(bet_fractions))],
        ]

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def valid_mask(self, obs: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=np.float32)
        valid_env = obs[indices.VALID_ACTIONS]  # [CHECK, FOLD, BET, CALL]

        if valid_env[PlayerAction.FOLD.value] == 1:
            mask[0] = 1
        if valid_env[PlayerAction.CHECK.value] == 1:
            mask[1] = 1
        if valid_env[PlayerAction.CALL.value] == 1:
            mask[2] = 1

        if valid_env[PlayerAction.BET.value] == 1:
            bet_low = float(obs[indices.VALID_BET_LOW])
            bet_high = float(obs[indices.VALID_BET_HIGH])
            if bet_high >= bet_low and bet_high > 0:
                mask[3:] = 1
        return mask

    def to_env_action(self, obs: np.ndarray, action_idx: int) -> Action:
        discrete = self.actions[action_idx]
        if discrete.kind == "FOLD":
            return Action(PlayerAction.FOLD, 0.0)
        if discrete.kind == "CHECK":
            return Action(PlayerAction.CHECK, 0.0)
        if discrete.kind == "CALL":
            return Action(PlayerAction.CALL, 0.0)

        bet_low = float(obs[indices.VALID_BET_LOW])
        bet_high = float(obs[indices.VALID_BET_HIGH])
        if bet_high <= bet_low:
            return Action(PlayerAction.BET, bet_low)

        frac = float(np.clip(self.bet_fractions[discrete.bucket_i], 0.0, 1.0))
        amount = bet_low + frac * (bet_high - bet_low)
        return Action(PlayerAction.BET, float(np.round(amount, 2)))


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class DeepCFRBackbone(nn.Module):
    """Residual MLP encoder for poker information-state features."""

    def __init__(self, obs_size: int, hidden_size: int, num_blocks: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(obs)
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


class AdvantageNet(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden_size: int, num_blocks: int, dropout: float):
        super().__init__()
        self.backbone = DeepCFRBackbone(obs_size, hidden_size, num_blocks, dropout)
        self.head = nn.Linear(hidden_size, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(obs))


class StrategyNet(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden_size: int, num_blocks: int, dropout: float):
        super().__init__()
        self.backbone = DeepCFRBackbone(obs_size, hidden_size, num_blocks, dropout)
        self.head = nn.Linear(hidden_size, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(obs))


@dataclass
class RegretSample:
    obs: np.ndarray
    advantages: np.ndarray
    valid_mask: np.ndarray


@dataclass
class StrategySample:
    obs: np.ndarray
    probs: np.ndarray
    valid_mask: np.ndarray


class ReservoirBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.data: List[object] = []
        self.seen = 0

    def __len__(self):
        return len(self.data)

    def add(self, item):
        self.seen += 1
        if len(self.data) < self.capacity:
            self.data.append(item)
            return
        replace_idx = int(self.rng.integers(0, self.seen))
        if replace_idx < self.capacity:
            self.data[replace_idx] = item

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, len(self.data), size=batch_size)
        return [self.data[i] for i in idx]


class _ListBuffer:
    """Simple append-only buffer used inside traversal worker processes."""

    def __init__(self):
        self.data = []

    def add(self, item):
        self.data.append(item)


def _split_work(total: int, parts: int) -> List[int]:
    parts = max(1, parts)
    base = total // parts
    rem = total % parts
    chunks = [base + (1 if i < rem else 0) for i in range(parts)]
    return [c for c in chunks if c > 0]


def _cpu_state_dict(module: nn.Module) -> dict:
    # Serialize as NumPy to avoid PyTorch FD-based tensor sharing overhead
    # when dispatching many multiprocessing tasks.
    return {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}


def _build_checkpoint(
    advantage_nets: List[AdvantageNet],
    advantage_opts: List[torch.optim.Optimizer],
    strategy_net: StrategyNet,
    strategy_opt: torch.optim.Optimizer,
    bet_fractions: List[float],
    args,
    iteration: int,
) -> dict:
    return {
        "iteration": iteration,
        "advantage_nets": [net.state_dict() for net in advantage_nets],
        "advantage_opts": [opt.state_dict() for opt in advantage_opts],
        "strategy_net": strategy_net.state_dict(),
        "strategy_opt": strategy_opt.state_dict(),
        "bet_fractions": bet_fractions,
        "players": args.players,
        "hidden_size": args.hidden_size,
        "num_blocks": args.num_blocks,
        "dropout": args.dropout,
    }


def _iteration_checkpoint_path(base_output: str, iteration: int) -> str:
    base = Path(base_output)
    return str(base.with_name(f"{base.stem}_iter{iteration:04d}{base.suffix}"))


def _run_traversal_worker(
    traverser_id: int,
    traversals: int,
    worker_seed: int,
    players: int,
    stack_low: int,
    stack_high: int,
    invalid_action_penalty: float,
    bet_fractions: List[float],
    hidden_size: int,
    num_blocks: int,
    dropout: float,
    advantage_state_dicts_cpu: List[dict],
    max_depth: int,
):
    # Keep each worker single-threaded to avoid oversubscription when many workers run.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    rng = np.random.default_rng(worker_seed)

    abstraction = ActionAbstraction(bet_fractions=bet_fractions)
    advantage_nets = [
        AdvantageNet(OBS_SIZE, abstraction.n_actions, hidden_size, num_blocks, dropout).to(device)
        for _ in range(players)
    ]
    for i in range(players):
        state_dict = {
            k: torch.from_numpy(v.copy())
            for k, v in advantage_state_dicts_cpu[i].items()
        }
        advantage_nets[i].load_state_dict(state_dict)
        advantage_nets[i].eval()

    regret_buffers = [_ListBuffer() for _ in range(players)]
    strategy_buffer = _ListBuffer()

    for _ in range(traversals):
        env = Table(
            n_players=players,
            stack_low=stack_low,
            stack_high=stack_high,
            invalid_action_penalty=invalid_action_penalty,
        )
        env.seed(int(worker_seed + rng.integers(0, 1_000_000)))
        obs = env.reset()
        traverse_from_obs(
            env=env,
            obs=obs,
            traverser_id=traverser_id,
            abstraction=abstraction,
            advantage_nets=advantage_nets,
            regret_buffers=regret_buffers,
            strategy_buffer=strategy_buffer,
            rng=rng,
            device=device,
            depth_left=max_depth,
        )

    return traverser_id, regret_buffers[traverser_id].data, strategy_buffer.data


def regret_matching(advantages: np.ndarray, valid_mask: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    positive = np.maximum(advantages, 0.0) * valid_mask
    total = positive.sum()
    if total <= eps:
        denom = valid_mask.sum()
        return valid_mask / max(denom, 1.0)
    return positive / total


@torch.no_grad()
def policy_from_adv_net(net: AdvantageNet, obs: np.ndarray, valid_mask: np.ndarray, device: torch.device) -> np.ndarray:
    was_training = net.training
    net.eval()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    advantages = net(obs_t).squeeze(0).cpu().numpy()
    if was_training:
        net.train()
    return regret_matching(advantages, valid_mask)


def traverse_from_obs(
    env: Table,
    obs: np.ndarray,
    traverser_id: int,
    abstraction: ActionAbstraction,
    advantage_nets: List[AdvantageNet],
    regret_buffers: List[ReservoirBuffer],
    strategy_buffer: ReservoirBuffer,
    rng: np.random.Generator,
    device: torch.device,
    depth_left: int,
) -> float:
    if depth_left <= 0:
        return 0.0

    acting_id = int(obs[indices.ACTING_PLAYER])
    valid_mask = abstraction.valid_mask(obs)
    policy = policy_from_adv_net(advantage_nets[acting_id], obs, valid_mask, device)

    strategy_buffer.add(StrategySample(obs=obs.copy(), probs=policy.copy(), valid_mask=valid_mask.copy()))

    valid_actions = np.flatnonzero(valid_mask > 0)
    if len(valid_actions) == 0:
        return 0.0

    if acting_id == traverser_id:
        action_values = np.zeros(abstraction.n_actions, dtype=np.float32)
        node_value = 0.0
        for action_idx in valid_actions:
            child_env = env.clone()
            env_action = abstraction.to_env_action(obs, int(action_idx))
            obs2, rewards, done, _ = child_env.step(env_action)
            utility = float(rewards[traverser_id]) if done else traverse_from_obs(
                child_env,
                obs2,
                traverser_id,
                abstraction,
                advantage_nets,
                regret_buffers,
                strategy_buffer,
                rng,
                device,
                depth_left - 1,
            )
            action_values[action_idx] = utility
            node_value += float(policy[action_idx] * utility)

        regret_buffers[traverser_id].add(
            RegretSample(
                obs=obs.copy(),
                advantages=(action_values - node_value).astype(np.float32),
                valid_mask=valid_mask.copy(),
            )
        )
        return node_value

    sampled_idx = int(rng.choice(valid_actions, p=policy[valid_actions] / policy[valid_actions].sum()))
    env_action = abstraction.to_env_action(obs, sampled_idx)
    obs2, rewards, done, _ = env.step(env_action)
    if done:
        return float(rewards[traverser_id])
    return traverse_from_obs(
        env,
        obs2,
        traverser_id,
        abstraction,
        advantage_nets,
        regret_buffers,
        strategy_buffer,
        rng,
        device,
        depth_left - 1,
    )


def train_advantage_net(
    net: AdvantageNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReservoirBuffer,
    device: torch.device,
    batch_size: int,
    train_steps: int,
    grad_clip: float,
    log_every: int = 0,
    log_prefix: str = "",
):
    if len(buffer) < batch_size:
        return
    net.train()
    for step_i in range(1, train_steps + 1):
        batch = buffer.sample(batch_size)
        obs = torch.tensor(np.stack([item.obs for item in batch]), dtype=torch.float32, device=device)
        target_advantages = torch.tensor(np.stack([item.advantages for item in batch]), dtype=torch.float32, device=device)
        valid_mask = torch.tensor(np.stack([item.valid_mask for item in batch]), dtype=torch.float32, device=device)

        pred_advantages = net(obs)
        loss = ((pred_advantages - target_advantages) ** 2 * valid_mask).sum(dim=1).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        if log_every > 0 and (step_i % log_every == 0 or step_i == train_steps):
            print(f"{log_prefix}adv_train step={step_i}/{train_steps} loss={loss.item():.5f}")


def train_strategy_net(
    net: StrategyNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReservoirBuffer,
    device: torch.device,
    batch_size: int,
    train_steps: int,
    grad_clip: float,
    log_every: int = 0,
    log_prefix: str = "",
):
    if len(buffer) < batch_size:
        return
    net.train()
    for step_i in range(1, train_steps + 1):
        batch = buffer.sample(batch_size)
        obs = torch.tensor(np.stack([item.obs for item in batch]), dtype=torch.float32, device=device)
        target_probs = torch.tensor(np.stack([item.probs for item in batch]), dtype=torch.float32, device=device)
        valid_mask = torch.tensor(np.stack([item.valid_mask for item in batch]), dtype=torch.float32, device=device)

        logits = net(obs)
        masked_logits = logits.masked_fill(valid_mask == 0, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        loss = -(target_probs * log_probs).sum(dim=1).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        if log_every > 0 and (step_i % log_every == 0 or step_i == train_steps):
            print(f"{log_prefix}strategy_train step={step_i}/{train_steps} loss={loss.item():.5f}")


def build_env_factory(args, rng: np.random.Generator) -> Callable[[], Table]:
    def make_env() -> Table:
        env = Table(
            n_players=args.players,
            stack_low=args.stack_low,
            stack_high=args.stack_high,
            invalid_action_penalty=args.invalid_action_penalty,
        )
        env.seed(int(args.seed + rng.integers(0, 1_000_000)))
        return env

    return make_env


def train(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    # In Linux/WSL multiprocessing, this avoids exhausting file descriptors
    # when many worker tasks pass tensors.
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (CUDA not available or --cpu was set).")

    bet_fractions = [float(x) for x in args.bet_fractions.split(",") if x.strip()]
    abstraction = ActionAbstraction(bet_fractions=bet_fractions)
    make_env = build_env_factory(args, rng)

    advantage_nets = [
        AdvantageNet(OBS_SIZE, abstraction.n_actions, args.hidden_size, args.num_blocks, args.dropout).to(device)
        for _ in range(args.players)
    ]
    advantage_opts = [torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay) for net in advantage_nets]

    strategy_net = StrategyNet(OBS_SIZE, abstraction.n_actions, args.hidden_size, args.num_blocks, args.dropout).to(device)
    strategy_opt = torch.optim.Adam(strategy_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    regret_buffers = [ReservoirBuffer(args.regret_buffer_capacity, rng) for _ in range(args.players)]
    strategy_buffer = ReservoirBuffer(args.strategy_buffer_capacity, rng)
    start_iteration = 1

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        ckpt_players = int(checkpoint.get("players", args.players))
        if ckpt_players != args.players:
            raise ValueError(
                f"--players ({args.players}) does not match checkpoint players ({ckpt_players}) in {args.resume_from}"
            )
        ckpt_bets = list(checkpoint.get("bet_fractions", bet_fractions))
        if ckpt_bets != bet_fractions:
            raise ValueError(
                f"--bet-fractions ({bet_fractions}) does not match checkpoint bet_fractions ({ckpt_bets})"
            )
        advantage_states = checkpoint["advantage_nets"]
        if len(advantage_states) != len(advantage_nets):
            raise ValueError("Checkpoint advantage_nets count does not match current player count.")

        for i in range(args.players):
            advantage_nets[i].load_state_dict(advantage_states[i])
        strategy_net.load_state_dict(checkpoint["strategy_net"])

        if "advantage_opts" in checkpoint and len(checkpoint["advantage_opts"]) == len(advantage_opts):
            for i in range(args.players):
                advantage_opts[i].load_state_dict(checkpoint["advantage_opts"][i])
        if "strategy_opt" in checkpoint:
            strategy_opt.load_state_dict(checkpoint["strategy_opt"])

        loaded_iter = int(checkpoint.get("iteration", 0))
        start_iteration = loaded_iter + 1
        print(f"Resumed from {args.resume_from} at iteration={loaded_iter}. Continuing at iteration={start_iteration}.")

    for iteration in range(start_iteration, args.iterations + 1):
        iter_start = time.perf_counter()
        if args.log_phase_timing:
            print(f"[iter {iteration}] start")

        traversals_start = time.perf_counter()
        if args.num_workers > 1:
            tasks = []
            advantage_state_dicts_cpu = [_cpu_state_dict(net) for net in advantage_nets]
            chunks_per_traverser = max(1, args.workers_per_traverser)
            for traverser_id in range(args.players):
                work_chunks = _split_work(args.traversals_per_player, chunks_per_traverser)
                for chunk_i, chunk in enumerate(work_chunks):
                    worker_seed = int(
                        args.seed
                        + iteration * 1_000_000
                        + traverser_id * 10_000
                        + chunk_i * 97
                    )
                    tasks.append((traverser_id, chunk, worker_seed))

            max_workers = min(args.num_workers, len(tasks))
            if args.log_phase_timing:
                print(
                    f"[iter {iteration}] multiprocessing traversals start "
                    f"(workers={max_workers}, tasks={len(tasks)})"
                )

            completed = 0
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = []
                for traverser_id, chunk, worker_seed in tasks:
                    futures.append(
                        pool.submit(
                            _run_traversal_worker,
                            traverser_id=traverser_id,
                            traversals=chunk,
                            worker_seed=worker_seed,
                            players=args.players,
                            stack_low=args.stack_low,
                            stack_high=args.stack_high,
                            invalid_action_penalty=args.invalid_action_penalty,
                            bet_fractions=bet_fractions,
                            hidden_size=args.hidden_size,
                            num_blocks=args.num_blocks,
                            dropout=args.dropout,
                            advantage_state_dicts_cpu=advantage_state_dicts_cpu,
                            max_depth=args.max_depth,
                        )
                    )

                for future in as_completed(futures):
                    traverser_id, regret_samples, strategy_samples = future.result()
                    for sample in regret_samples:
                        regret_buffers[traverser_id].add(sample)
                    for sample in strategy_samples:
                        strategy_buffer.add(sample)
                    completed += 1
                    if args.log_mp_every > 0 and (completed % args.log_mp_every == 0 or completed == len(futures)):
                        print(
                            f"[iter {iteration}] mp_tasks={completed}/{len(futures)} "
                            f"(regret+={len(regret_samples)}, strategy+={len(strategy_samples)})"
                        )
        else:
            for traverser_id in range(args.players):
                if args.log_phase_timing:
                    print(f"[iter {iteration}] traverser={traverser_id} traversals start")
                for _ in range(args.traversals_per_player):
                    traversal_i = _ + 1
                    env = make_env()
                    obs = env.reset()
                    traverse_from_obs(
                        env=env,
                        obs=obs,
                        traverser_id=traverser_id,
                        abstraction=abstraction,
                        advantage_nets=advantage_nets,
                        regret_buffers=regret_buffers,
                        strategy_buffer=strategy_buffer,
                        rng=rng,
                        device=device,
                        depth_left=args.max_depth,
                    )
                    if args.log_traversal_every > 0 and (traversal_i % args.log_traversal_every == 0 or traversal_i == args.traversals_per_player):
                        print(
                            f"[iter {iteration}] traverser={traverser_id} "
                            f"traversals={traversal_i}/{args.traversals_per_player}"
                        )
        traversals_s = time.perf_counter() - traversals_start

        adv_start = time.perf_counter()
        for pid in range(args.players):
            train_advantage_net(
                net=advantage_nets[pid],
                optimizer=advantage_opts[pid],
                buffer=regret_buffers[pid],
                device=device,
                batch_size=args.batch_size,
                train_steps=args.adv_train_steps,
                grad_clip=args.grad_clip,
                log_every=args.log_train_every,
                log_prefix=f"[iter {iteration}] [player {pid}] ",
            )
        adv_s = time.perf_counter() - adv_start

        strat_start = time.perf_counter()
        train_strategy_net(
            net=strategy_net,
            optimizer=strategy_opt,
            buffer=strategy_buffer,
            device=device,
            batch_size=args.batch_size,
            train_steps=args.strategy_train_steps,
            grad_clip=args.grad_clip,
            log_every=args.log_train_every,
            log_prefix=f"[iter {iteration}] ",
        )
        strat_s = time.perf_counter() - strat_start
        iter_s = time.perf_counter() - iter_start

        if iteration % args.log_every == 0:
            regret_sizes = [len(buf) for buf in regret_buffers]
            print(
                f"iter={iteration:4d} regret_bufs={regret_sizes} "
                f"strategy_buf={len(strategy_buffer)} "
                f"time_s(total={iter_s:.1f}, traversals={traversals_s:.1f}, adv={adv_s:.1f}, strategy={strat_s:.1f})"
            )

        if args.save_every > 0 and iteration % args.save_every == 0:
            ckpt = _build_checkpoint(
                advantage_nets=advantage_nets,
                advantage_opts=advantage_opts,
                strategy_net=strategy_net,
                strategy_opt=strategy_opt,
                bet_fractions=bet_fractions,
                args=args,
                iteration=iteration,
            )
            periodic_path = _iteration_checkpoint_path(args.output, iteration)
            torch.save(ckpt, periodic_path)
            torch.save(ckpt, args.output)
            print(f"Saved periodic checkpoint to {periodic_path} (and updated {args.output})")

    checkpoint = _build_checkpoint(
        advantage_nets=advantage_nets,
        advantage_opts=advantage_opts,
        strategy_net=strategy_net,
        strategy_opt=strategy_opt,
        bet_fractions=bet_fractions,
        args=args,
        iteration=max(start_iteration - 1, args.iterations),
    )
    torch.save(checkpoint, args.output)
    print(f"Saved Deep CFR checkpoint to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Deep CFR-style trainer for no-limit hold'em using pokerenv")
    parser.add_argument("--iterations", type=int, default=100, help="Number of CFR iterations")
    parser.add_argument("--traversals-per-player", type=int, default=100, help="Traversals per player each iteration")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Worker processes for traversal generation (1 disables multiprocessing)",
    )
    parser.add_argument(
        "--workers-per-traverser",
        type=int,
        default=2,
        help="How many traversal tasks to split each traverser into per iteration",
    )
    parser.add_argument("--players", type=int, default=6, choices=[2, 3, 4, 5, 6], help="Players at the table")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden layer width")
    parser.add_argument("--num-blocks", type=int, default=6, help="Number of residual blocks in each network")
    parser.add_argument("--dropout", type=float, default=0.10, help="Dropout in residual blocks")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--adv-train-steps", type=int, default=200, help="Advantage net SGD steps per iteration")
    parser.add_argument("--strategy-train-steps", type=int, default=200, help="Strategy net SGD steps per iteration")
    parser.add_argument("--regret-buffer-capacity", type=int, default=200000, help="Reservoir capacity per regret buffer")
    parser.add_argument("--strategy-buffer-capacity", type=int, default=500000, help="Reservoir capacity for strategy samples")
    parser.add_argument("--bet-fractions", type=str, default="0.1,0.25,0.5,0.75,1.0", help="Comma-separated bet bucket fractions in [0,1]")
    parser.add_argument("--max-depth", type=int, default=512, help="Traversal depth cutoff")
    parser.add_argument("--stack-low", type=int, default=50, help="Minimum stack in big blinds")
    parser.add_argument("--stack-high", type=int, default=200, help="Maximum stack in big blinds")
    parser.add_argument("--invalid-action-penalty", type=float, default=0.01, help="Penalty for invalid actions")
    parser.add_argument("--log-every", type=int, default=1, help="How often to print logs (iterations)")
    parser.add_argument(
        "--log-traversal-every",
        type=int,
        default=25,
        help="Print traversal progress every N traversals per traverser (0 disables)",
    )
    parser.add_argument(
        "--log-train-every",
        type=int,
        default=50,
        help="Print training progress every N SGD steps (0 disables)",
    )
    parser.add_argument(
        "--log-phase-timing",
        action="store_true",
        help="Print iteration phase start markers for easier long-run monitoring",
    )
    parser.add_argument(
        "--log-mp-every",
        type=int,
        default=2,
        help="Print multiprocessing traversal task completion every N tasks (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="deep_cfr_checkpoint.pt", help="Path for the trained checkpoint")
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save periodic checkpoints every N iterations (0 disables periodic saves)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Resume training from an existing checkpoint path",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
