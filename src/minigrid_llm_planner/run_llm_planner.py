import argparse
import gymnasium as gym
from .core.agent import MinigridLLMAgent
from .core.llm_planner import LLMPlanner
from .core.random_goal_env import RandomGoalEmptyEnv
import csv
import os

def register_environments(agent_view_size: int):
    """Register custom environments with Gymnasium."""
    try:
        gym.register(
            id='MiniGrid-RandomGoalEmpty-8x8-v0',
            entry_point='src.minigrid_llm_planner.core.random_goal_env:RandomGoalEmptyEnv',
            kwargs={'size': 8, 'agent_view_size': agent_view_size}
        )
    except Exception as e:
        print(f"Environment might already be registered: {e}")

def main(args):
    """Main script for Minigrid LLM Agent inference."""
    # Create environment
    if args.env == 'random_goal_empty':
        env = RandomGoalEmptyEnv(size=8, render_mode="human" if args.render else None, agent_view_size=args.view_size)
    else:
        # Only register if we're using gym.make()
        register_environments(agent_view_size=args.view_size)
        try:
            env = gym.make(
                args.env,
                render_mode="human" if args.render else None,
                agent_view_size=args.view_size
            )
        except gym.error.Error as e:
            print(f"Error creating environment: {e}")
            raise
    
    # Create planner
    planner = LLMPlanner(model_name=args.model, api_key=args.api_key, temperature=args.temperature)

    # Create agent. 
    agent = MinigridLLMAgent(
        env=env,
        planner=planner,
    )

    # Use the provided results directory directly
    results_dir = args.results_dir
    
    # Create and write to files directly in the results directory
    csv_filename = os.path.join(results_dir, "results.csv")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Episode', 'Success', 'Steps'])

        total_success = 0
        total_steps = 0

        for episode in range(args.episodes):
            episode_num = episode + 1
            print(f"\nEpisode {episode_num}/{args.episodes}")
            print("-" * 50)

            # Create episode-specific log file directly in results_dir
            episode_log = os.path.join(results_dir, f"episode_{episode_num:03d}.log")
            with open(episode_log, 'w') as log_file:
                log_file.write(f"Episode {episode_num}/{args.episodes}\n")
                log_file.write("-" * 50 + "\n")

                if args.seed is not None:
                    curr_seed = args.seed + episode
                else:
                    curr_seed = None

                # Run episode with the seed
                episode_info = agent.run_episode(max_steps=args.max_steps, seed=curr_seed)

                # Update statistics
                total_success += int(episode_info['success'])
                total_steps += episode_info['steps']

                # Write to CSV
                csvwriter.writerow([episode_num, episode_info['success'], episode_info['steps']])

                # Write episode results to log file
                log_file.write(f"\nEpisode Results:\n")
                log_file.write(f"Success: {episode_info['success']}\n")
                log_file.write(f"Steps Taken: {episode_info['steps']}\n")

                # Write last reasoning for debugging
                if episode_info['plans']:
                    log_file.write("\nLast Reasoning:\n")
                    log_file.write(f"{episode_info['plans'][-1]}\n")

        # Write summary directly to results_dir
        summary_file = os.path.join(results_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write("\nOverall Results:\n")
            f.write("-" * 50 + "\n")
            success_rate = total_success / args.episodes * 100
            avg_steps = total_steps / args.episodes
            f.write(f"Success Rate: {success_rate:.2f}%\n")
            f.write(f"Average Steps: {avg_steps:.2f}\n")

    print(f"\nResults saved to {results_dir}")

    # Cleanup
    agent.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Minigrid LLM Agent')
    parser.add_argument('--env', type=str, default='random_goal_empty', 
                        help='Environment name')
    parser.add_argument('--view_size', type=int, default=5,
                        help='View size')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='LLM model name')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=25,
                      help='Maximum steps per episode')
    parser.add_argument('--render', action='store_true',
                      help='Enable environment rendering', default=True)
    parser.add_argument('--api-key', type=str, help='API key for OpenAI')
    parser.add_argument('--seed', type=int, help='Random seed for environment')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for LLM')
    parser.add_argument('--results-dir', type=str, required=True, help='Directory to save results')
    args = parser.parse_args()
    
    main(args)
