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

    def generate_plan(self, state_description: str, history: list) -> str:
        """Generate next action based on current state observation."""
        try:
            # Format history into a readable string
            history_context = ""
            if history:
                history_context = "Previous steps:\n"
                for entry in history[-5:]:  # Last 5 steps
                    history_context += f"Step {entry['step']}:\n"
                    history_context += f"State: {entry['state']}\n"
                    history_context += f"Action taken: {entry['action']}\n\n"

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
            action = parts[-1].lower()  # Last line contains the action

            print(f"Reasoning: {reasoning}")
            print(f"Generated action: {action}")
            
            # Validate and store action
            if action in self.valid_actions:
                return action
            return "forward"  # Default to forward if invalid response

        except Exception as e:
            print(f"Error in planning: {str(e)}")
            return "forward"  # Default to forward
