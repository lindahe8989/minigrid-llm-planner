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
        self.last_abstract_state = None
        self.door_seen = False
        self.door_direction = None  # 'east', 'west', etc.
        self.has_key = False
        self.door_position = None  # When door is seen
        self.key_position = None   # When key is seen
        self.explored_positions = set()  # Track where we've been
        self.last_action = None
        self.door_opened = False # Track if the door has been opened 

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

    def _get_abstract_state(self, state_desc):
        """Enhanced state abstraction with door/key priority"""
        abstract_states = []
        
        # Check for key/door in view
        if 'key' in state_desc.lower():
            abstract_states.append("key_visible")
        if 'door' in state_desc.lower():
            self.door_seen = True
            abstract_states.append("door_visible")
            
        # Basic position state
        if 'wall directly ahead' in state_desc.lower():
            abstract_states.append("at_wall")
        else:
            abstract_states.append("path_clear")
            
        return abstract_states

    def generate_plan(self, state_description: str, history: list) -> str:
        """Generate next action based on current state observation."""
        try:
            # Get current position from history
            current_pos = history[-1]['position'] if history else None
            
            # Update exploration memory
            if current_pos:
                hit_wall = "wall" in state_description.lower()
                self._update_exploration_memory(current_pos, hit_wall)

            # Check if the door is opened 
            if "unlocked door" in state_description.lower():
                self.door_opened = True 

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
            # print("\n=== Historical Context ===")
            # print(history_context)
            print("=== Current State ===")
            print(state_description)
            print("=====================\n")

            # Get abstract state
            abstract_state = self._get_abstract_state(state_description)
            
            # Detect if stuck in same abstract state
            if self.last_abstract_state == abstract_state and "at_wall" in abstract_state:
                print("Stuck in same abstract state, changing strategy...")
                self.last_abstract_state = abstract_state
                return "right"  # Simple escape strategy
                
            self.last_abstract_state = abstract_state
            
            # Include abstract state in prompt
            prompt = f"""
            === Current State ===
            {state_description}
            Abstract state: {', '.join(abstract_state)}
            =====================
            
            You are an agent trying to navigate through a grid world. Your goal is to reach the target/goal, but you need to be strategic:
            1. Sometimes, you will need to get a key to open a door in order to open up new paths and see the goal. 
            2. To do so, you must first find the key and then use the key to open the door. 
            3. If you haven't gotten the key when you see the door, remember the door's location and continue exploring.
            4. The state description will tell you if you've already unlocked a door. If you have:
               - Focus on exploring new areas beyond the door to find the goal
               - Don't waste time looking for more keys or checking other doors
               - The goal is likely in an area you couldn't access before
            5. If you haven't unlocked any doors yet:
               - Prioritize finding a key if you don't have one
               - If you have a key, prioritize finding and unlocking a door
            6. The state description will tell you the location of the door/key relative to you.
            7. If you have a key and you see the door, you should first move towards the door, and then toggle it. 
            If the door is one step away from you (like one step right or one step left but never one step right and one step left), you should change your direction to face the door (unless you are already facing it). 
            If you are directly facing it and one step away, you should just toggle the door instead of moving toward it.
            8. Just because a path is clear doesn't mean you should take it - think about your goal
            
            What action should I take? Consider the door/key first before deciding to move forward.
            """
            
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
                {"role": "user", "content": prompt}
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
            
            # If the door is opened, focus on reaching the goal 
            if self.door_opened:
                print("Door is opened, focusing on reaching the goal...")
                # Implement logic to prioritize reaching the goal
                # For example, if the path is clear, move forward
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
            # If the door is unlocked, focus on exploration
            if "already unlocked a door" in state_description:
                print("Door is unlocked, focusing on reaching the goal...")
                # Prioritize forward movement and exploration of new areas
                if "path ahead is clear" in state_description.lower():
                    return "forward"
                    
            return action
            return "forward"  # Default to forward if invalid response

        except Exception as e:
            print(f"Error in planning: {str(e)}")
            return "forward"

    def get_current_goal(self):
        if not self.has_key and self.key_position:
            return "reach_key"
        elif self.has_key and self.door_position:
            return "reach_door"
        elif not self.has_key and not self.key_position:
            return "explore_for_key"
        else:
            return "explore_for_door"

    def _analyze_view(self, view):
        """Extract key information from the agent's view matrix"""
        info = {
            'has_key': self.has_key,
            'door_visible': False,
            'door_location': None,
            'can_interact_with_door': False
        }
        
        # Convert view to string for analysis
        view_str = str(view)
        
        # Look for door in view
        if 'D' in view_str:
            info['door_visible'] = True
            # Additional door position analysis...
            
        return info

    def _analyze_state(self, state_desc, view):
        """Analyze state and view to extract key information"""
        info = {
            'has_key': self.has_key,
            'door_visible': False,
            'can_open_door': False,
            'at_door': False
        }
        
        # Update key status if we just picked up a key
        if self.last_action == 'pickup' and 'key' in state_desc.lower():
            self.has_key = True
            
        # Look for door in view matrix
        if 'D' in str(view):
            info['door_visible'] = True
            # Check if we're right at the door
            if 'door directly ahead' in state_desc.lower():
                info['at_door'] = True
                if self.has_key:
                    info['can_open_door'] = True
                    
        return info

    def plan_action(self, state_desc, history):
        # Update key status based on previous actions
        if self.last_action == 'pickup' and 'key' in state_desc.lower():
            self.has_key = True
            
        view_info = self._analyze_view(history[-1].get('view', []))
        
        prompt = f"""
        === Current State ===
        {state_desc}
        
        Important Information:
        - Have key: {self.has_key}
        - Door visible: {view_info['door_visible']}
        - Door location: {view_info['door_location']}
        
        Priority Rules:
        1. If you see a key and don't have one, getting it is top priority
        2. If you have a key AND see a door, then going to the door is priority
        3. If you see a door but don't have a key, IGNORE the door and continue exploring
        4. When exploring, prefer moving forward and turning only when blocked
        
        What single action should you take? Think step by step.
        """
        
        # Force toggle if at door with key
        if view_info['can_interact_with_door'] and self.has_key:
            return 'toggle'
            
        action = self._get_action_from_llm(prompt)
        self.last_action = action
        return action
