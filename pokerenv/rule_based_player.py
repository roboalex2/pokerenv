from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from treys import Card, Deck, Evaluator

from pokerenv.common import Action, GameState, PlayerAction


RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = {1: "c", 2: "d", 4: "h", 8: "s"}


@dataclass
class RuleBotConfig:
    monte_carlo_samples: int = 300
    bluff_freq: float = 0.05
    rng_seed: int = 7


class RuleBasedPlayer:
    """Strong heuristic baseline for pokerenv.

    Policy:
    - Preflop: hand-strength and position-aware opening/3-bet logic.
    - Postflop: Monte Carlo equity vs random opponent ranges + pot odds.
    - Bet sizing: pot-fraction based, clipped to env-valid range.
    """

    def __init__(self, config: RuleBotConfig | None = None):
        self.cfg = config or RuleBotConfig()
        self.rng = np.random.default_rng(self.cfg.rng_seed)
        self.evaluator = Evaluator()

    def get_action(self, obs: np.ndarray) -> Action:
        valid = self._valid_actions(obs)
        to_call = max(float(obs[21] - obs[14]), 0.0)
        pot = max(float(obs[20]), 1e-6)
        street = int(obs[15])
        bet_low = float(obs[5])
        bet_high = float(obs[6])
        can_bet = PlayerAction.BET in valid and bet_high >= bet_low and bet_high > 0

        if street == GameState.PREFLOP.value:
            strength = self._preflop_strength(obs)
            return self._decide_preflop(
                strength=strength,
                to_call=to_call,
                pot=pot,
                valid=valid,
                can_bet=can_bet,
                bet_low=bet_low,
                bet_high=bet_high,
                obs=obs,
            )

        equity = self._postflop_equity(obs)
        return self._decide_postflop(
            equity=equity,
            to_call=to_call,
            pot=pot,
            valid=valid,
            can_bet=can_bet,
            bet_low=bet_low,
            bet_high=bet_high,
            obs=obs,
        )

    def _decide_preflop(
        self,
        strength: float,
        to_call: float,
        pot: float,
        valid: List[PlayerAction],
        can_bet: bool,
        bet_low: float,
        bet_high: float,
        obs: np.ndarray,
    ) -> Action:
        position = int(obs[7])
        opps = self._active_opponents(obs)
        pos_bonus = 0.04 if position >= 4 else (0.02 if position >= 2 else 0.0)
        strength += pos_bonus

        if to_call <= 0:
            if can_bet and strength >= 0.72:
                return Action(PlayerAction.BET, self._sized_bet(0.75, pot, bet_low, bet_high))
            if can_bet and strength >= 0.60 and position >= 3:
                return Action(PlayerAction.BET, self._sized_bet(0.55, pot, bet_low, bet_high))
            if can_bet and strength >= 0.52 and self.rng.random() < self.cfg.bluff_freq:
                return Action(PlayerAction.BET, self._sized_bet(0.45, pot, bet_low, bet_high))
            return self._safe_check_or_fold(valid)

        pot_odds = to_call / (pot + to_call)
        call_edge = strength - pot_odds

        if call_edge < -0.10 and PlayerAction.FOLD in valid:
            return Action(PlayerAction.FOLD, 0.0)

        raise_threshold = 0.76 - 0.03 * min(opps, 4)
        if can_bet and strength >= raise_threshold:
            return Action(PlayerAction.BET, self._sized_bet(0.95, pot + to_call, bet_low, bet_high))

        if call_edge >= -0.02 and PlayerAction.CALL in valid:
            return Action(PlayerAction.CALL, 0.0)

        if PlayerAction.FOLD in valid:
            return Action(PlayerAction.FOLD, 0.0)
        return self._safe_check_or_fold(valid)

    def _decide_postflop(
        self,
        equity: float,
        to_call: float,
        pot: float,
        valid: List[PlayerAction],
        can_bet: bool,
        bet_low: float,
        bet_high: float,
        obs: np.ndarray,
    ) -> Action:
        street = int(obs[15])
        opps = self._active_opponents(obs)
        # Multiway discount.
        equity_adj = equity - 0.03 * max(opps - 1, 0)

        if to_call <= 0:
            if can_bet and equity_adj >= 0.72:
                frac = 0.85 if street == GameState.RIVER.value else 0.70
                return Action(PlayerAction.BET, self._sized_bet(frac, pot, bet_low, bet_high))
            if can_bet and equity_adj >= 0.58:
                return Action(PlayerAction.BET, self._sized_bet(0.55, pot, bet_low, bet_high))
            if can_bet and equity_adj < 0.45 and self.rng.random() < self.cfg.bluff_freq:
                return Action(PlayerAction.BET, self._sized_bet(0.45, pot, bet_low, bet_high))
            return self._safe_check_or_fold(valid)

        pot_odds = to_call / (pot + to_call)
        margin = equity_adj - pot_odds
        strong_raise = 0.74 if street <= GameState.TURN.value else 0.70

        if margin < -0.06 and PlayerAction.FOLD in valid:
            return Action(PlayerAction.FOLD, 0.0)

        if can_bet and equity_adj >= strong_raise:
            frac = 1.00 if street == GameState.RIVER.value else 0.80
            return Action(PlayerAction.BET, self._sized_bet(frac, pot + to_call, bet_low, bet_high))

        if margin >= -0.01 and PlayerAction.CALL in valid:
            return Action(PlayerAction.CALL, 0.0)

        if PlayerAction.FOLD in valid:
            return Action(PlayerAction.FOLD, 0.0)
        return self._safe_check_or_fold(valid)

    def _safe_check_or_fold(self, valid: List[PlayerAction]) -> Action:
        if PlayerAction.CHECK in valid:
            return Action(PlayerAction.CHECK, 0.0)
        if PlayerAction.FOLD in valid:
            return Action(PlayerAction.FOLD, 0.0)
        if PlayerAction.CALL in valid:
            return Action(PlayerAction.CALL, 0.0)
        return Action(PlayerAction.CHECK, 0.0)

    def _valid_actions(self, obs: np.ndarray) -> List[PlayerAction]:
        valid = []
        valid_vec = obs[1:5]
        if valid_vec[PlayerAction.CHECK.value] == 1:
            valid.append(PlayerAction.CHECK)
        if valid_vec[PlayerAction.FOLD.value] == 1:
            valid.append(PlayerAction.FOLD)
        if valid_vec[PlayerAction.BET.value] == 1:
            valid.append(PlayerAction.BET)
        if valid_vec[PlayerAction.CALL.value] == 1:
            valid.append(PlayerAction.CALL)
        return valid

    def _active_opponents(self, obs: np.ndarray) -> int:
        # Others block starts at index 23 and uses 6 fields each.
        # state field offset=1, all_in offset=5.
        count = 0
        for i in range(5):
            base = 23 + i * 6
            state = int(obs[base + 1])
            all_in = int(obs[base + 5])
            if state == 1 or all_in == 1:
                count += 1
        return max(count, 1)

    def _preflop_strength(self, obs: np.ndarray) -> float:
        r1 = int(obs[9])
        r2 = int(obs[11])
        s1 = int(obs[8])
        s2 = int(obs[10])
        hi, lo = max(r1, r2), min(r1, r2)
        paired = hi == lo
        suited = s1 == s2
        gap = hi - lo

        # Base from high card quality.
        score = 0.12 + 0.045 * hi + 0.02 * lo
        if paired:
            score += 0.24 + 0.025 * hi
        if suited:
            score += 0.05
        if gap == 1:
            score += 0.05
        elif gap == 2:
            score += 0.02
        elif gap >= 4:
            score -= 0.04

        # Premium broadway / ace-x suited boosts.
        if hi == 12 and lo >= 8:
            score += 0.05
        if hi == 12 and suited:
            score += 0.03

        return float(np.clip(score, 0.0, 1.0))

    def _postflop_equity(self, obs: np.ndarray) -> float:
        hole, board = self._cards_from_obs(obs)
        used_cards = set(hole + board)
        if len(hole) != 2:
            return 0.5

        opps = self._active_opponents(obs)
        need_board = 5 - len(board)
        wins = 0.0
        total = 0.0

        full = Deck.GetFullDeck()
        remain = [c for c in full if c not in used_cards]
        if len(remain) < opps * 2 + need_board:
            return 0.5

        for _ in range(self.cfg.monte_carlo_samples):
            self.rng.shuffle(remain)
            ptr = 0
            opp_holes = []
            for _j in range(opps):
                opp_holes.append([remain[ptr], remain[ptr + 1]])
                ptr += 2
            sim_board = list(board)
            for _j in range(need_board):
                sim_board.append(remain[ptr])
                ptr += 1

            hero_rank = self.evaluator.evaluate(hole, sim_board)
            opp_ranks = [self.evaluator.evaluate(opp, sim_board) for opp in opp_holes]
            best_opp = min(opp_ranks)
            if hero_rank < best_opp:
                wins += 1.0
            elif hero_rank == best_opp:
                ties = sum(1 for r in opp_ranks if r == hero_rank) + 1
                wins += 1.0 / ties
            total += 1.0

        return wins / max(total, 1.0)

    def _cards_from_obs(self, obs: np.ndarray) -> Tuple[List[int], List[int]]:
        hole = []
        board = []
        c1 = self._obs_card_to_int(int(obs[8]), int(obs[9]))
        c2 = self._obs_card_to_int(int(obs[10]), int(obs[11]))
        if c1 is not None:
            hole.append(c1)
        if c2 is not None:
            hole.append(c2)
        for i in range(5):
            c = self._obs_card_to_int(int(obs[16 + i * 2]), int(obs[17 + i * 2]))
            if c is not None:
                board.append(c)
        return hole, board

    def _obs_card_to_int(self, suit_i: int, rank_i: int) -> int | None:
        if suit_i == 0:
            return None
        suit = SUIT_CHARS.get(suit_i)
        if suit is None or not (0 <= rank_i <= 12):
            return None
        rank = RANK_CHARS[rank_i]
        return Card.new(rank + suit)

    def _sized_bet(self, pot_frac: float, pot: float, low: float, high: float) -> float:
        target = pot * pot_frac
        amount = float(np.clip(target, low, high))
        return float(np.round(amount, 2))
