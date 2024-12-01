from openai import OpenAI


class LLMPlanner:
    """LLM based planner for Minigrid environments."""

    def __init__(self, api_key: str, model_name: str, temperature: float = 0.0):
        """Initialize the LLM planner."""
        self.model_name = model_name
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.temperature = temperature
        # Valid actions that match the agent's action_mapping
        self.valid_actions = {'forward', 'left', 'right', 'pickup', 'drop', 'toggle', 'done'}
        
        # Add exploration memory
        self.visited_zones = {}  # (x,y) -> visit_count
        self.dead_ends = set()   # Set of (x,y) positions where agent got stuck
        self.visited_positions = {}  # (x,y) -> {direction: count}
        self.movement_history = []   # List of (pos, direction) tuples
        self.unexplored_directions = set()  # Tracks which directions haven't been tried at each position

    def _update_exploration_memory(self, position: tuple, hit_wall: bool = False):
        """Update visit counts and dead ends."""
        # Update visit count
        self.visited_zones[position] = self.visited_zones.get(position, 0) + 1
        
        # Mark dead ends when hitting walls or getting stuck
        if hit_wall or self.visited_zones[position] > 3:  # If visited too many times
            self.dead_ends.add(position)

    def _analyze_movement_patterns(self, current_pos, current_direction, state_description):
        """Analyze movement history to make smarter decisions."""
        # Update visit counts for this position-direction combination
        pos_key = tuple(current_pos)
        if pos_key not in self.visited_positions:
            self.visited_positions[pos_key] = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.visited_positions[pos_key][current_direction] += 1
        
        # Add to movement history
        self.movement_history.append((pos_key, current_direction))
        
        # If we've tried this direction too many times at this position
        if self.visited_positions[pos_key][current_direction] > 2:
            # Find directions we haven't tried much at this position
            less_visited = [d for d in ['north', 'south', 'east', 'west'] 
                           if self.visited_positions[pos_key][d] < 2]
            
            # If we've been moving vertically a lot, prefer horizontal directions
            recent_moves = [d for _, d in self.movement_history[-5:]]
            if all(d in ['north', 'south'] for d in recent_moves):
                horizontal = [d for d in less_visited if d in ['east', 'west']]
                if horizontal:
                    return self._get_turn_action(current_direction, horizontal[0])
                
            # If we've been moving horizontally a lot, prefer vertical directions
            if all(d in ['east', 'west'] for d in recent_moves):
                vertical = [d for d in less_visited if d in ['north', 'south']]
                if vertical:
                    return self._get_turn_action(current_direction, vertical[0])
        
        return None  # No override needed

    def _get_turn_action(self, current_dir, target_dir):
        """Get the action needed to turn from current_dir to target_dir."""
        # Map of what direction you end up in after turning left/right
        turns = {
            'north': {'left': 'west', 'right': 'east'},
            'south': {'left': 'east', 'right': 'west'},
            'east': {'left': 'north', 'right': 'south'},
            'west': {'left': 'south', 'right': 'north'}
        }
        
        # Try left turn
        if turns[current_dir]['left'] == target_dir:
            return 'left'
        # Try right turn
        if turns[current_dir]['right'] == target_dir:
            return 'right'
        # Need to turn around
        return 'right'  # Will take two turns

    def generate_plan(self, state_description: str, history: list) -> str:
        """Generate next action based on current state observation."""
        try:
            # Get current position from history
            current_pos = history[-1]['position'] if history else None
            
            # Update exploration memory
            if current_pos:
                hit_wall = "wall" in state_description.lower()
                self._update_exploration_memory(current_pos, hit_wall)

            # Format history into a readable string
            history_context = ""
            if history:
                history_context = "Previous steps:\n"
                for entry in history[-5:]:  # Last 5 steps
                    history_context += f"Step {entry['step']}:\n"
                    history_context += f"State: {entry['state']}\n"
                    history_context += f"Action taken: {entry['action']}\n\n"
                    history_context += f"Agent Position: {entry['position']}\n\n"

            # Print the context
            print("\n=== Historical Context ===")
            print(history_context)
            print("=== Current State ===")
            print(state_description)
            print("=====================\n")

            system_prompt = f"""You are a navigation agent in a grid world. Your task is to reach the goal (G).
Available actions are:
- 'forward': Move one step forward (You cannot move forward if you are facing a wall)
- 'left': Turn left 90 degrees
- 'right': Turn right 90 degrees
- 'toggle': Open doors or interact with objects
- 'pickup': Pick up an object
- 'drop': Drop the carried object
- 'done': Complete the mission

{history_context}
Current state:
{state_description}

Move efficiently towards the goal. Note that you cannot walk into walls. Further note that the grid you can see is only part of the entire environment. Your field of vision is only 5x5 but the environment is much larger.
First explain your reasoning, then respond with a single word action from the list above."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state_description}
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )

            full_response = response.choices[0].message.content.strip()
            
            # Split reasoning and action
            parts = full_response.split('\n')
            reasoning = '\n'.join(parts[:-1])  # All lines except the last one
            action = parts[-1].lower().replace('action:', '').strip()

            print(f"Reasoning: {reasoning}")
            print(f"Generated action: {action}")
            
            # First priority: If path is clear and action is forward, ALWAYS take it
            if "path ahead is clear" in state_description.lower() and action == 'forward':
                return action
            
            # Check for repeated patterns in last 2 steps
            if len(history) >= 2:
                last_two_actions = [entry['action'] for entry in history[-2:]]
                last_two_positions = [entry['position'] for entry in history[-2:]]
                print("last two actions", last_two_actions)
                print("last two positions", last_two_positions)
                
                # Check if both actions and positions are repeating
                if (last_two_actions[0] == last_two_actions[1] and 
                    last_two_positions[0] == last_two_positions[1] and 
                    action == last_two_actions[0]):
                    print("Detected action loop! Changing default behavior...")
                    # If we're stuck in forward/backward, try turning
                    if action in ['forward']:
                        return 'right'
                    # If we're stuck turning, try moving
                    elif action in ['left', 'right']:
                        return 'forward'
                    # For any other repeated action, try turning right
                    else:
                        return 'right'
            return action
            return "forward"  # Default to forward if invalid response

        except Exception as e:
            print(f"Error in planning: {str(e)}")
            return "forward"
