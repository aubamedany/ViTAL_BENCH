#!/bin/bash
#SBATCH -J deepseek6          # Job name
#SBATCH -o %x-%j.out                  # Standard output log
#SBATCH -e %x-%j.err                  # Standard error log
#SBATCH -p GPU-1               # Partition for A40 GPU nodes[cite: 2]
#SBATCH -n 1                          # Number of CPU cores[cite: 2]
#SBATCH --gres=gpu:1                  # Request 1 GPU[cite: 2]
#SBATCH --mem=32g                     # Request 32GB system memory[cite: 2]

# Ensure we are in the directory where you submitted the job[cite: 2]
cd ${SLURM_SUBMIT_DIR}

# Load the CUDA module (required for GPU processing)[cite: 2]
module load cuda
module load java/21
# Initialize Conda for this bash session
# Note: 'conda activate' usually requires the shell to be initialized[cite: 2]
source $(conda info --base)/etc/profile.d/conda.sh
conda activate traffic

python -m agent.cli --condition e0_oracle --llm vlm --model internvl-1b --out logs/agent/e0_oracle_new/internvl-1b.json
python -m agent.cli --condition e0_oracle --llm vlm --model internvl-2b --out logs/agent/e0_oracle_new/internvl-2b.json
python -m agent.cli --condition e0_oracle --llm vlm --model internvl-8b --out logs/agent/e0_oracle_new/internvl-8b.json
python -m agent.cli --condition e0_oracle --llm vlm --model internvl-9b --out logs/agent/e0_oracle_new/internvl-9b.json

python -m agent.cli --condition e0_oracle --llm vlm --model qwen3vl-2b --out logs/agent/e0_oracle_new/qwen3vl-2b.json
python -m agent.cli --condition e0_oracle --llm vlm --model qwen3vl-4b --out logs/agent/e0_oracle_new/qwen3vl-4b.json
python -m agent.cli --condition e0_oracle --llm vlm --model qwen3vl-8b --out logs/agent/e0_oracle_new/qwen3vl-8b.json