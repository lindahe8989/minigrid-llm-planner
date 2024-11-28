# Minigrid LLM Planner

A baseline implementation of trajectory planning using GPT-4 for Minigrid environments. This project demonstrates how to use large language models for planning in simple grid-world environments.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/minigrid-llm-planner.git
   cd minigrid-llm-planner
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

5. **Make scripts executable:**
   ```bash
   chmod +x scripts/run_baseline_experiments.sh
   ```

## Running Experiments

Run the baseline experiments:
```bash
bash scripts/run_baseline_experiments.sh
```

Results are saved in timestamped directories with the following structure:
```
results/
└── baseline_experiments_<timestamp>/
    ├── experiment_config.txt
    ├── <environment_name>/
    │   ├── stdout/
    │   │   └── episode_*.log
    │   ├── episode_*.log
    │   ├── results.csv
    │   └── summary.txt
```