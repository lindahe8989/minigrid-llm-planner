#!/bin/bash

# Change to project root directory (assuming script is in scripts/)
cd "$(dirname "$0")/.." || exit

# Create timestamped directory for this experiment run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="results/baseline_experiments_${TIMESTAMP}"
mkdir -p "$EXPERIMENT_DIR"

# Set common parameters
SEED=42
MODEL="gpt-4o-mini"
EPISODES=10
VIEW_SIZE=5
MAX_STEPS=50
TEMPERATURE=0.2

# Log experiment parameters
echo "Experiment Parameters:" > "${EXPERIMENT_DIR}/experiment_config.txt"
echo "Model: ${MODEL}" >> "${EXPERIMENT_DIR}/experiment_config.txt"
echo "Episodes: ${EPISODES}" >> "${EXPERIMENT_DIR}/experiment_config.txt"
echo "View Size: ${VIEW_SIZE}" >> "${EXPERIMENT_DIR}/experiment_config.txt"
echo "Max Steps: ${MAX_STEPS}" >> "${EXPERIMENT_DIR}/experiment_config.txt"
echo "Temperature: ${TEMPERATURE}" >> "${EXPERIMENT_DIR}/experiment_config.txt"
echo "Seed: ${SEED}" >> "${EXPERIMENT_DIR}/experiment_config.txt"
echo "Timestamp: ${TIMESTAMP}" >> "${EXPERIMENT_DIR}/experiment_config.txt"

echo "Starting baseline experiments..."
echo "Results will be saved in: ${EXPERIMENT_DIR}"

# Run experiment on MiniGrid-Empty-8x8-v0
for episode in $(seq 1 $EPISODES); do
    episode_padded=$(printf "%03d" $episode)
    echo "Running experiments on MiniGrid-Empty-8x8-v0 - Episode $episode_padded..."
    episode_dir="${EXPERIMENT_DIR}/MiniGrid-Empty-8x8-v0/episode_${episode_padded}"
    mkdir -p "$episode_dir"
    python -m src.minigrid_llm_planner.run_llm_planner \
        --env "MiniGrid-Empty-8x8-v0" \
        --model $MODEL \
        --episodes 1 \
        --view_size $VIEW_SIZE \
        --max-steps $MAX_STEPS \
        --temperature $TEMPERATURE \
        --seed $SEED \
        --results-dir "$episode_dir" \
        2>&1 | tee "${episode_dir}/stdout.log"
done

# Run experiment on random_goal_empty
for episode in $(seq 1 $EPISODES); do
    episode_padded=$(printf "%03d" $episode)
    echo -e "\nRunning experiments on random_goal_empty - Episode $episode_padded..."
    episode_dir="${EXPERIMENT_DIR}/random_goal_empty/episode_${episode_padded}"
    mkdir -p "$episode_dir"
    python -m src.minigrid_llm_planner.run_llm_planner \
        --env "random_goal_empty" \
        --model $MODEL \
        --episodes 1 \
        --view_size $VIEW_SIZE \
        --max-steps $MAX_STEPS \
        --temperature $TEMPERATURE \
        --seed $SEED \
        --results-dir "$episode_dir" \
        2>&1 | tee "${episode_dir}/stdout.log"
done

echo "Baseline experiments completed!"
echo "Results are saved in: ${EXPERIMENT_DIR}"