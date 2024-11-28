from minigrid.envs.empty import EmptyEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal

class RandomGoalEmptyEnv(EmptyEnv):
    """
    Empty environment with randomly placed goal
    """
    def __init__(
        self,
        size=8,
        render_mode=None,
        agent_view_size=5
    ):
        super().__init__(
            size=size,
            render_mode=render_mode,
            agent_view_size=agent_view_size
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Fill outer walls
        self.grid.wall_rect(0, 0, width, height)

        # Place agent at random position
        self.place_agent()

        # Place goal at random position
        self.place_obj(Goal())