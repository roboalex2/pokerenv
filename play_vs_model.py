"""Interactive poker runner for testing a trained model checkpoint.

Modes:
- hu: 1 human vs 1 model
- ring_human_vs_5_model: 1 human + 5 model players
- ring_human_vs_1_model_4_random: 1 human + 1 model + 4 random players
- ring_model_vs_5_humans: 1 model + 5 human-controlled opponents
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

import pokerenv.obs_indices as indices
from pokerenv.common import Action, PlayerAction
from pokerenv.table import Table
from self_play_train import OBS_SIZE, ActionAbstraction, StrategyNet


RANK_STR = {
    0: "2",
    1: "3",
    2: "4",
    3: "5",
    4: "6",
    5: "7",
    6: "8",
    7: "9",
    8: "T",
    9: "J",
    10: "Q",
    11: "K",
    12: "A",
}
SUIT_STR = {1: "c", 2: "d", 4: "h", 8: "s"}
ACTION_NAME = {
    PlayerAction.CHECK: "check",
    PlayerAction.FOLD: "fold",
    PlayerAction.CALL: "call",
    PlayerAction.BET: "bet",
}


def card_str(suit_i: int, rank_i: int) -> str:
    if suit_i in SUIT_STR and rank_i in RANK_STR:
        return f"{RANK_STR[rank_i]}{SUIT_STR[suit_i]}"
    return "??"


def board_from_obs(obs: np.ndarray) -> str:
    cards = []
    for i in range(5):
        suit_i = int(obs[16 + i * 2])
        rank_i = int(obs[17 + i * 2])
        if suit_i == 0 and rank_i == 0:
            continue
        cards.append(card_str(suit_i, rank_i))
    return " ".join(cards) if cards else "-"


def hand_from_obs(obs: np.ndarray) -> str:
    c1 = card_str(int(obs[8]), int(obs[9]))
    c2 = card_str(int(obs[10]), int(obs[11]))
    return f"{c1} {c2}"


def valid_actions_from_obs(obs: np.ndarray) -> List[PlayerAction]:
    valid = []
    valid_vec = obs[indices.VALID_ACTIONS]
    if valid_vec[PlayerAction.CHECK.value] == 1:
        valid.append(PlayerAction.CHECK)
    if valid_vec[PlayerAction.FOLD.value] == 1:
        valid.append(PlayerAction.FOLD)
    if valid_vec[PlayerAction.BET.value] == 1:
        valid.append(PlayerAction.BET)
    if valid_vec[PlayerAction.CALL.value] == 1:
        valid.append(PlayerAction.CALL)
    return valid


def describe_action(action: Action) -> str:
    if action.action_type is PlayerAction.BET:
        return f"bet {action.bet_amount:.2f}"
    return ACTION_NAME[action.action_type]


@dataclass
class Controller:
    label: str

    def act(self, obs: np.ndarray, acting_id: int) -> Action:
        raise NotImplementedError


class HumanController(Controller):
    def act(self, obs: np.ndarray, acting_id: int) -> Action:
        valid_actions = valid_actions_from_obs(obs)
        to_call = max(float(obs[21] - obs[14]), 0.0)
        bet_low = float(obs[indices.VALID_BET_LOW])
        bet_high = float(obs[indices.VALID_BET_HIGH])

        print()
        print(f"[{self.label}] id={acting_id} hand={hand_from_obs(obs)} board={board_from_obs(obs)}")
        print(
            f"pot={obs[indices.POT_SIZE]:.2f} stack={obs[indices.ACTING_PLAYER_STACK_SIZE]:.2f} "
            f"to_call={to_call:.2f}"
        )
        if PlayerAction.BET in valid_actions:
            print(f"bet range: [{bet_low:.2f}, {bet_high:.2f}]")

        menu: List[Tuple[int, PlayerAction]] = []
        for i, action in enumerate(valid_actions, start=1):
            menu.append((i, action))
            print(f"{i}) {ACTION_NAME[action]}")

        while True:
            choice_raw = input("Choose action number: ").strip()
            try:
                choice = int(choice_raw)
            except ValueError:
                print("Please enter a valid number.")
                continue

            matches = [a for i, a in menu if i == choice]
            if not matches:
                print("Action number out of range.")
                continue
            action_type = matches[0]

            if action_type is PlayerAction.BET:
                while True:
                    amount_raw = input("Enter bet amount: ").strip().lower()
                    if amount_raw == "min":
                        amount = bet_low
                    elif amount_raw == "max" or amount_raw == "allin":
                        amount = bet_high
                    else:
                        try:
                            amount = float(amount_raw)
                        except ValueError:
                            print("Enter a number, or 'min'/'max'.")
                            continue
                    if amount < bet_low or amount > bet_high:
                        print(f"Amount must be in [{bet_low:.2f}, {bet_high:.2f}].")
                        continue
                    return Action(PlayerAction.BET, float(np.round(amount, 2)))
            return Action(action_type, 0.0)


class RandomController(Controller):
    def __init__(self, label: str, rng: np.random.Generator):
        super().__init__(label=label)
        self.rng = rng

    def act(self, obs: np.ndarray, acting_id: int) -> Action:
        valid_actions = valid_actions_from_obs(obs)
        chosen = valid_actions[int(self.rng.integers(0, len(valid_actions)))]
        if chosen is PlayerAction.BET:
            low = float(obs[indices.VALID_BET_LOW])
            high = float(obs[indices.VALID_BET_HIGH])
            amount = float(np.round(self.rng.uniform(low, high), 2))
            return Action(PlayerAction.BET, amount)
        return Action(chosen, 0.0)


class ModelController(Controller):
    def __init__(
        self,
        label: str,
        net: StrategyNet,
        abstraction: ActionAbstraction,
        rng: np.random.Generator,
        device: torch.device,
    ):
        super().__init__(label=label)
        self.net = net
        self.abstraction = abstraction
        self.rng = rng
        self.device = device

    @torch.no_grad()
    def act(self, obs: np.ndarray, acting_id: int) -> Action:
        self.net.eval()
        valid_mask = self.abstraction.valid_mask(obs)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.net(obs_t).squeeze(0)
        mask_t = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
        masked_logits = logits.masked_fill(~mask_t, -1e9)
        probs = torch.softmax(masked_logits, dim=-1).cpu().numpy()
        valid_idx = np.flatnonzero(valid_mask > 0)

        if probs[valid_idx].sum() <= 0 or np.any(np.isnan(probs[valid_idx])):
            probs = np.zeros_like(probs)
            probs[valid_idx] = 1.0 / len(valid_idx)
        else:
            probs[valid_idx] /= probs[valid_idx].sum()

        action_idx = int(self.rng.choice(valid_idx, p=probs[valid_idx]))
        return self.abstraction.to_env_action(obs, action_idx)


def load_model(
    checkpoint_path: str,
    device: torch.device,
    seed: int,
) -> Tuple[StrategyNet, ActionAbstraction, np.random.Generator]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_obs_size = int(checkpoint.get("obs_size", OBS_SIZE))
    if ckpt_obs_size != OBS_SIZE:
        raise ValueError(
            f"Checkpoint obs_size ({ckpt_obs_size}) does not match current environment obs_size ({OBS_SIZE})."
        )
    bet_fractions = checkpoint["bet_fractions"]
    players = int(checkpoint.get("players", 6))
    hidden_size = int(checkpoint["hidden_size"])
    history_hidden = int(checkpoint.get("history_hidden", 128))
    num_blocks = int(checkpoint["num_blocks"])
    dropout = float(checkpoint["dropout"])

    abstraction = ActionAbstraction(bet_fractions=bet_fractions)
    strategy_net = StrategyNet(
        OBS_SIZE,
        abstraction.n_actions,
        hidden_size=hidden_size,
        history_hidden=history_hidden,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)
    strategy_net.load_state_dict(checkpoint["strategy_net"])
    strategy_net.eval()

    print(
        f"Loaded checkpoint '{checkpoint_path}' "
        f"(trained_players={players}, actions={abstraction.n_actions})"
    )
    return strategy_net, abstraction, np.random.default_rng(seed)


def build_controllers(
    mode: str,
    n_players: int,
    hero_id: int,
    model_id: int,
    model_controller: ModelController,
    rng: np.random.Generator,
) -> Dict[int, Controller]:
    if hero_id >= n_players or model_id >= n_players:
        raise ValueError("hero_id/model_id must be within [0, n_players-1].")

    controllers: Dict[int, Controller] = {}
    if mode == "hu":
        if n_players != 2:
            raise ValueError("hu mode requires n_players=2.")
        other_id = 1 if hero_id == 0 else 0
        controllers[hero_id] = HumanController(label="You")
        controllers[other_id] = model_controller
        return controllers

    if mode == "ring_human_vs_5_model":
        for pid in range(n_players):
            controllers[pid] = HumanController(label="You") if pid == hero_id else model_controller
        return controllers

    if mode == "ring_human_vs_1_model_4_random":
        if hero_id == model_id:
            raise ValueError("hero_id and model_id must be different in this mode.")
        for pid in range(n_players):
            if pid == hero_id:
                controllers[pid] = HumanController(label="You")
            elif pid == model_id:
                controllers[pid] = model_controller
            else:
                controllers[pid] = RandomController(label=f"Random-{pid}", rng=rng)
        return controllers

    if mode == "ring_model_vs_5_humans":
        for pid in range(n_players):
            if pid == model_id:
                controllers[pid] = model_controller
            else:
                controllers[pid] = HumanController(label=f"You (controls id {pid})")
        return controllers

    raise ValueError(f"Unknown mode: {mode}")


def print_hand_header(env: Table, controllers: Dict[int, Controller], hand_no: int) -> None:
    print()
    print("=" * 72)
    print(f"Hand {hand_no}")
    for seat, player in enumerate(env.players, start=1):
        label = controllers[player.identifier].label
        print(
            f"seat={seat} id={player.identifier} role={label} "
            f"stack={player.stack:.2f} pos={player.position}"
        )
    print("=" * 72)


def parse_args():
    parser = argparse.ArgumentParser(description="Play against a trained poker model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint from self_play_train.py")
    parser.add_argument(
        "--mode",
        type=str,
        default="hu",
        choices=[
            "hu",
            "ring_human_vs_5_model",
            "ring_human_vs_1_model_4_random",
            "ring_model_vs_5_humans",
        ],
        help="Game mode",
    )
    parser.add_argument("--hands", type=int, default=10, help="Number of hands to play")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for model inference")
    parser.add_argument("--hero-id", type=int, default=0, help="Human-controlled player id")
    parser.add_argument("--model-id", type=int, default=1, help="Model-controlled player id")
    parser.add_argument("--stack-low", type=int, default=50, help="Minimum stack in big blinds")
    parser.add_argument("--stack-high", type=int, default=200, help="Maximum stack in big blinds")
    return parser.parse_args()


def main():
    args = parse_args()
    n_players = 2 if args.mode == "hu" else 6
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    strategy_net, abstraction, rng = load_model(args.checkpoint, device=device, seed=args.seed)
    model_controller = ModelController(
        label="Model",
        net=strategy_net,
        abstraction=abstraction,
        rng=rng,
        device=device,
    )
    controllers = build_controllers(
        mode=args.mode,
        n_players=n_players,
        hero_id=args.hero_id,
        model_id=args.model_id,
        model_controller=model_controller,
        rng=rng,
    )

    env = Table(
        n_players=n_players,
        stack_low=args.stack_low,
        stack_high=args.stack_high,
        invalid_action_penalty=0.0,
    )
    env.seed(args.seed)

    cumulative = np.zeros(n_players, dtype=np.float64)
    for hand_no in range(1, args.hands + 1):
        obs = env.reset()
        print_hand_header(env, controllers, hand_no)

        done = False
        while not done:
            acting_id = int(obs[indices.ACTING_PLAYER])
            controller = controllers[acting_id]
            action = controller.act(obs, acting_id)
            print(f"id={acting_id} [{controller.label}] -> {describe_action(action)}")
            obs, rewards, done, _ = env.step(action)

        rewards = rewards.astype(np.float64)
        cumulative += rewards
        reward_line = " ".join([f"id {pid}: {rewards[pid]:+.2f}" for pid in range(n_players)])
        cum_line = " ".join([f"id {pid}: {cumulative[pid]:+.2f}" for pid in range(n_players)])
        print(f"Hand {hand_no} result: {reward_line}")
        print(f"Cumulative: {cum_line}")

    print()
    print("Session complete.")


if __name__ == "__main__":
    main()
