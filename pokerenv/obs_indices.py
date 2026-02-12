ACTING_PLAYER = 0
VALID_ACTIONS = [*range(1, 5)]
VALID_BET_LOW = 5
VALID_BET_HIGH = 6
ACTING_PLAYER_POSITION = 7
ACTING_PLAYER_STACK_SIZE = 12
POT_SIZE = 20

# Base scalar observation size from the original environment layout.
BASE_OBS_SIZE = 58

# Encoded action-history sequence appended after BASE_OBS_SIZE.
# Each event is:
# [valid, is_check, is_fold, is_bet, is_call, position_norm, street_norm, log1p_amount]
HISTORY_LEN = 24
HISTORY_EVENT_SIZE = 8
HISTORY_START = BASE_OBS_SIZE
HISTORY_END = HISTORY_START + HISTORY_LEN * HISTORY_EVENT_SIZE

# Full observation size.
OBS_SIZE = HISTORY_END
