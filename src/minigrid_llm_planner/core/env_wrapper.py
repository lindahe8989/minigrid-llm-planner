from typing import Dict, Any, Tuple, List, Optional
from numpy.typing import NDArray
import gymnasium as gym
import numpy as np

from src.minigrid_llm_planner.core.constants import IDX_TO_OBJECT, OBJECT_TO_IDX, OBJECT_SYMBOLS, DIRECTION_TEXT, VIEW_OFFSET, Direction


class MinigridEnvWrapper(gym.Wrapper):
    """A wrapper for Minigrid environments that provides additional functionality.
    
    This wrapper adds:
    - Full grid state tracking
    - ASCII visualization
    - Natural language state descriptions
    """
    
    def __init__(self, env: gym.Env):
        """Initialize the wrapper.
        
        Args:
            env: The Minigrid environment to wrap
        """
        super().__init__(env)
        self.last_obs: Optional[Dict[str, Any]] = None
        
        # Initialize grid size from environment
        env_unwrapped = env.unwrapped
        self.grid_size = (env_unwrapped.width, env_unwrapped.height)
        
        # Initialize full grid with correct size
        self.full_grid = np.zeros((self.grid_size[1], self.grid_size[0], 3), dtype=np.uint8)

    def _transform_grid_by_direction(self, grid: NDArray, agent_dir: int) -> NDArray:
        """Transform grid based on agent's direction.
        
        Args:
            grid: The grid to transform
            agent_dir: The agent's current direction (0-3)
            
        Returns:
            The transformed grid
        """
        if agent_dir == 1:  # facing down
            return np.rot90(np.flip(grid, axis=1), k=-1)
        elif agent_dir == 3:  # facing up
            return np.rot90(np.flip(grid, axis=1), k=-3)
        elif agent_dir == 2:  # facing left
            return np.flip(grid, axis=0)
        elif agent_dir == 0:  # facing right
            return np.flip(grid, axis=1)
        return grid

    def _get_agent_position_and_arrow(self, agent_dir: int, view_width: int) -> Tuple[Tuple[int, int], str]:
        """Get agent position and direction arrow based on agent direction.
        
        Args:
            agent_dir: The agent's current direction (0-3)
            view_width: Width of the view grid
            
        Returns:
            Tuple of ((x, y), arrow_symbol)
        """
        if agent_dir == 0:    # facing right
            return (2, 0), '→'
        elif agent_dir == 1:  # facing down
            return (0, 2), '↓'
        elif agent_dir == 2:  # facing left
            return (2, 4), '←'
        else:                 # facing up
            return (4, 2), '↑'

    def _get_goal_description(self, grid: Any, center_x: int, agent_y: int, agent_dir: int) -> Optional[str]:
        """Generate description of goal location relative to agent.
        
        Args:
            grid: The environment grid
            center_x: X coordinate of center
            agent_y: Y coordinate of agent
            agent_dir: Agent's current direction
            
        Returns:
            Description string or None if goal not found
        """
        for j in range(grid.height):
            for i in range(grid.width):
                cell = grid.get(i, j)
                if cell and cell.type == "goal":
                    return self._generate_relative_position(
                        i, j, center_x, agent_y, agent_dir == 1
                    )
        return None

    def _generate_agent_position_in_grid(self) -> Tuple[int, int]:
        """Get the agent's absolute position in the grid.
        
        Returns:
            Tuple[int, int]: (x, y) coordinates of the agent in the grid
        """
        if self.last_obs is None:
            raise ValueError("No observation available. Did you call reset()?")
            
        env = self.env.unwrapped
        agent_pos = (int(env.agent_pos[0]), int(env.agent_pos[1]))
        
        print(f"Agent absolute position: {agent_pos}")
        return agent_pos

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = obs
        
        # Get environment instance
        env = self.env.unwrapped
        
        # Re-initialize full grid with zeros
        self.full_grid.fill(0)
        
        # Initialize or update the full grid
        self._update_full_grid(obs['image'], env.agent_pos, env.agent_dir)
        
        # Add visualization data to info dictionary
        info.update({
            'observer_view': self._create_grid_view(self.full_grid, env.agent_pos, env.agent_dir),
            'agent_view': self._create_agent_view_grid(),
            'state_description': self.get_state_description()
        })
        
        return obs, info

    def step(self, action: int):
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs
        
        # Get environment instance
        env = self.env.unwrapped
        
        # Update the full grid with new observation
        self._update_full_grid(obs['image'], env.agent_pos, env.agent_dir)
        
        # Add visualization data to info dictionary
        info.update({
            'observer_view': self._create_grid_view(self.full_grid, env.agent_pos, env.agent_dir),
            'agent_view': self._create_agent_view_grid(),
            'state_description': self.get_state_description()
        })
        
        return obs, reward, terminated, truncated, info

    def _update_full_grid(self, visible_grid: np.ndarray, agent_pos: tuple, agent_dir: int):
        """Update the full grid with the current observation."""
        view_height, view_width, _ = visible_grid.shape
        
        if view_height != view_width:
            raise ValueError(f"Expected square view, got {view_height}x{view_width}")
        
        view_size = view_height
        view_offset = VIEW_OFFSET
        
        # Update the full grid with the visible portion
        for i in range(view_size):
            for j in range(view_size):
                # Convert view coordinates to world coordinates based on agent's direction
                if agent_dir == Direction.RIGHT:
                    world_x = agent_pos[0] + j - view_offset
                    world_y = agent_pos[1] + i - view_offset
                elif agent_dir == Direction.DOWN:
                    world_x = agent_pos[0] - i + view_offset
                    world_y = agent_pos[1] + j - view_offset
                elif agent_dir == Direction.LEFT:
                    world_x = agent_pos[0] - j + view_offset
                    world_y = agent_pos[1] - i + view_offset
                elif agent_dir == Direction.UP:
                    world_x = agent_pos[0] + i - view_offset
                    world_y = agent_pos[1] - j + view_offset
                
                # Check if the world coordinates are within bounds
                if (0 <= world_x < self.grid_size[0] and 
                    0 <= world_y < self.grid_size[1]):
                    self.full_grid[world_y, world_x] = visible_grid[i, j]

    def _create_grid_view(self, grid: np.ndarray, agent_pos: tuple, agent_dir: int) -> str:
        """Create ASCII visualization of the agent's current view."""
        visible_grid = self.last_obs['image']
        view_height, view_width, _ = visible_grid.shape
        
        # Transform grid based on agent direction
        if agent_dir == Direction.DOWN:  # facing down
            visible_grid = np.flip(visible_grid, axis=1)
            visible_grid = np.rot90(visible_grid, k=-1)
        elif agent_dir == Direction.UP:  # facing up
            visible_grid = np.flip(visible_grid, axis=1)
            visible_grid = np.rot90(visible_grid, k=-3)
        elif agent_dir == Direction.LEFT:  # facing left
            visible_grid = np.flip(visible_grid, axis=0)
        elif agent_dir == Direction.RIGHT:  # facing right
            visible_grid = np.flip(visible_grid, axis=1)

        # Use OBJECT_SYMBOLS instead of hardcoding
        symbols = OBJECT_SYMBOLS
        
        # Use DIRECTION_SYMBOLS for agent position and arrow
        agent_pos, arrow = self._get_agent_position_and_arrow(agent_dir, view_width)
        
        rows = []
        for i in range(view_height):
            row = []
            for j in range(view_width):
                if (i, j) == agent_pos:
                    row.append(arrow)
                else:
                    cell = visible_grid[i, j]
                    if cell[0] == OBJECT_TO_IDX["floor"] and cell[1] == 5:
                        row.append(OBJECT_SYMBOLS["unseen"])
                    else:
                        row.append(symbols.get(IDX_TO_OBJECT.get(cell[0], "unknown"), str(cell[0])))
            rows.append(' '.join(row))
        
        return '\n'.join(rows)

    def _create_agent_view_grid(self) -> str:
        """Create a standardized string representation of the agent's view."""
        if self.last_obs is None:
            raise ValueError("No observation available. Did you call reset()?")

        env = self.env.unwrapped
        
        # Direction symbols
        dir_to_symbol = {
            0: "→",  # Facing right
            1: "↓",  # Facing down
            2: "←",  # Facing left
            3: "↑",  # Facing up
        }
        
        # Object type to symbol mapping
        obj_to_symbol = {
            "wall": "W",
            "floor": ".",
            "door": "D",
            "key": "K",
            "ball": "A",
            "box": "B",
            "goal": "G",
            "lava": "V",
        }
        
        # Get the agent's view grid
        grid, vis_mask = env.gen_obs_grid()
        
        view_lines = []
        for j in range(grid.height):
            line = []
            for i in range(grid.width):
                if not vis_mask[i, j]:
                    line.append("?")
                    continue
                    
                cell = grid.get(i, j)
                
                if (i, j) == (grid.width // 2, grid.height - 1):
                    line.append(dir_to_symbol[3])  # Agent always faces up
                elif cell is None:
                    line.append(".")
                else:
                    line.append(obj_to_symbol.get(cell.type, "?"))
            
            view_lines.append(" ".join(line))
        
        return "\n".join(view_lines)

    def get_state_description(self) -> str:
        """Convert current observation to text description."""
        if self.last_obs is None:
            raise ValueError("No observation available. Did you call reset()?")

        env = self.env.unwrapped
        agent_dir = DIRECTION_TEXT[Direction(env.agent_dir)]
        
        # Get the standardized grid view
        grid, vis_mask = env.gen_obs_grid()
        
        description = [f"You are facing {agent_dir}.\n"]
        
        # Describe what's directly ahead
        center_x = grid.width // 2
        agent_y = grid.height - 1
        front_cell = grid.get(center_x, agent_y - 1)
        
        if front_cell is None:
            description.append("The path ahead is clear.")
        else:
            description.append(f"There is a {front_cell.type} directly ahead.")
            
        # Look for goal
        for j in range(grid.height):
            for i in range(grid.width):
                cell = grid.get(i, j)
                if cell and cell.type == "goal":
                    # When facing south, the coordinate system is flipped 180 degrees
                    if env.agent_dir == Direction.DOWN:  # Facing south
                        if i < center_x and j < agent_y:
                            description.append("The goal is ahead and to the left.")  # Changed from right
                        elif i > center_x and j < agent_y:
                            description.append("The goal is ahead and to the right.")  # Changed from left
                        elif i < center_x:
                            description.append("The goal is to the left.")  # Changed from right
                        elif i > center_x:
                            description.append("The goal is to the right.")  # Changed from left
                        else:
                            description.append("The goal is directly ahead.")
                    else:
                        if i < center_x and j < agent_y:
                            description.append("The goal is ahead and to the left.")
                        elif i > center_x and j < agent_y:
                            description.append("The goal is ahead and to the right.")
                        elif i < center_x:
                            description.append("The goal is to the left.")
                        elif i > center_x:
                            description.append("The goal is to the right.")
                        else:
                            description.append("The goal is directly ahead.")
                    break
        
        return "\n".join(description)