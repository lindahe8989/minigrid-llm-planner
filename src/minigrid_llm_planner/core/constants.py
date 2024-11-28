from typing import Dict, Final
from enum import IntEnum

# Object mapping constants
OBJECT_TO_IDX: Final[Dict[str, int]] = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}

# Create reverse mapping from index to object name
IDX_TO_OBJECT: Final[Dict[int, str]] = {v: k for k, v in OBJECT_TO_IDX.items()}

# Direction enumeration
class Direction(IntEnum):
    RIGHT = 0  # Facing right
    DOWN = 1   # Facing down
    LEFT = 2   # Facing left
    UP = 3     # Facing up

# Direction to symbol mapping
DIRECTION_SYMBOLS: Final[Dict[Direction, str]] = {
    Direction.RIGHT: "→",
    Direction.DOWN: "↓",
    Direction.LEFT: "←",
    Direction.UP: "↑",
}

# Direction to text mapping
DIRECTION_TEXT: Final[Dict[Direction, str]] = {
    Direction.RIGHT: "east",
    Direction.DOWN: "south",
    Direction.LEFT: "west",
    Direction.UP: "north",
}

# Object visualization symbols
OBJECT_SYMBOLS: Final[Dict[str, str]] = {
    "unseen": "?",
    "empty": ".",
    "wall": "W",
    "floor": "F",
    "door": "D",
    "key": "K",
    "ball": "B",
    "box": "X",
    "goal": "G",
    "lava": "L",
    "agent": "A",
}

# Grid configuration
DEFAULT_GRID_SIZE: Final[int] = 5  # Default size for the agent's view (5x5)
VIEW_OFFSET: Final[int] = DEFAULT_GRID_SIZE // 2  # Offset from center to edge of view
