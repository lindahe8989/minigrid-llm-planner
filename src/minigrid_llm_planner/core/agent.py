import gymnasium as gym
from typing import Dict, Any, Optional
from minigrid.core.actions import Actions

from src.minigrid_llm_planner.core.llm_planner import LLMPlanner
from src.minigrid_llm_planner.core.env_wrapper import OBJECT_TO_IDX, MinigridEnvWrapper


class MinigridLLMAgent:
    """Main agent class that coordinates environment interaction and reasoning."""
    def __init__(
        self,
        env: gym.Env,
        planner: Optional[LLMPlanner] = None,
        verbose: bool = True,
        max_history_size: int = 10,
    ):
        """Initialize the agent with specified environment and model."""
        self.verbose = verbose
        self.env = env
        self.wrapped_env = MinigridEnvWrapper(self.env)
        if planner is None:
            raise ValueError("LLMPlanner instance required")
        self.planner = planner
        self.history = []  # List of (state_desc, action) tuples
        self.max_history_size = max_history_size

    def reset(self, seed: int | None = None) -> tuple[Any, Dict[str, Any]]:
        """Reset the environment and return initial observation."""
        if hasattr(self.planner, 'reset'):
            self.planner.reset()
        return self.wrapped_env.reset(seed=seed)

    def step(self, action: int) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Execute action in the environment."""
        obs, reward, terminated, truncated, info = self.wrapped_env.step(action)
        return obs, reward, terminated, truncated, info

    def run_episode(self, max_steps: int = 100, seed: int | None = None) -> Dict[str, Any]:
        """Run a single episode."""
        obs, info = self.reset(seed=seed)
        total_reward = 0
        steps_taken = 0
        consecutive_failures = 0
        reasoning_history = []
        terminated = truncated = False

        if self.verbose:
            print("\nInitial State:")
            self._print_current_state(info)
            print("\nStarting episode execution...")

        while not (terminated or truncated) and steps_taken < max_steps:
            if self.verbose:
                print(f"\n{'#' * 10} Step {steps_taken + 1} {'#' * 10}")
                self._print_current_state(info)
                print("\nObserving state and reasoning:")
            
            # Get current state description and generate action
            state_desc = self.wrapped_env.get_state_description()
            
            if self.verbose:
                print("\nObserving state and reasoning:")
                
            action_name = self.planner.generate_plan(state_desc, self.history)

            # Get agent's position from the environment wrapper
            position_in_grid = self.wrapped_env._generate_agent_position_in_grid()

            # Store state, action, and position in history
            self.history.append({
                'state': state_desc,
                'action': action_name,
                'step': steps_taken + 1,
                'position': position_in_grid  # Add position to history
            })
            
            # Keep history to a reasonable size (last N steps)
            if len(self.history) > self.max_history_size:
                self.history = self.history[-self.max_history_size:]

            if not action_name:
                if self.verbose:
                    print("No action returned from planner")
                consecutive_failures += 1
                continue

            # Store the reasoning for debugging
            reasoning_history.append(action_name)

            action_mapping = {
                'forward': Actions.forward,
                'left': Actions.left,
                'right': Actions.right,
                'pickup': Actions.pickup,
                'drop': Actions.drop,
                'toggle': Actions.toggle,
                'done': Actions.done
            }

            if action_name not in action_mapping:
                if self.verbose:
                    print(f"Invalid action: {action_name}")
                consecutive_failures += 1
                continue

            action = action_mapping[action_name].value
            if self.verbose:
                print(f"\nExecuting action: {action_name} (value: {action})")

            # Execute action and get feedback
            obs, reward, terminated, truncated, info = self.step(action)

            # Update statistics
            total_reward += reward
            steps_taken += 1

            if consecutive_failures >= 5:
                if self.verbose:
                    print("Too many failed actions, ending episode")
                break

            if self.verbose:
                print(f"Step result: reward={reward}, terminated={terminated}, truncated={truncated}")

        # Episode ended, return results
        success = total_reward > 0
        return {
            "success": success,
            "total_reward": total_reward,
            "steps": steps_taken,
            "plans": reasoning_history
        }

    def _print_current_state(self, info: Dict[str, Any]) -> None:
        """Print the current state of the environment."""
        if self.verbose:
            print("\nCurrent view (Observer POV):")
            print(info.get('observer_view', 'Observer view not available'))
            print("\nCurrent view (Agent POV):")
            print(info.get('agent_view', 'Agent view not available'))
            print("\nState description:")
            print(info.get('state_description', 'State description not available'))

    def close(self) -> None:
        """Clean up resources."""
        self.env.close()
