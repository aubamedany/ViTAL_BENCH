#!/bin/bash
#SBATCH -J deepseek6b_repoeval            # Job name
#SBATCH -o %x-%j.out                  # Standard output log
#SBATCH -e %x-%j.err                  # Standard error log
#SBATCH -p GPU-1                   # Partition for A40 GPU nodes[cite: 2]
#SBATCH -n 1                          # Number of CPU cores[cite: 2]
#SBATCH --gres=gpu:1                  # Request 1 GPU[cite: 2]
#SBATCH --mem=32g                     # Request 32GB system memory[cite: 2]

# Ensure we are in the directory where you submitted the job[cite: 2]
cd ${SLURM_SUBMIT_DIR}

# Load the CUDA module (required for GPU processing)[cite: 2]
module load cuda

# Initialize Conda for this bash session
# Note: 'conda activate' usually requires the shell to be initialized[cite: 2]
source $(conda info --base)/etc/profile.d/conda.sh
conda activate traffic

# Run the inference script
# python answer_vqa.py --model qwen3vl-2b --type law  --options 5
# python answer_vqa.py --model qwen3vl-2b --type law  --options 4
# python answer_vqa.py --model qwen3vl-2b --type detection  --options 5
# python answer_vqa.py --model qwen3vl-2b --type detection  --options 4


# python answer_vqa.py --model qwen3vl-4b --type law  --options 5
# python answer_vqa.py --model qwen3vl-4b --type law  --options 4
# python answer_vqa.py --model qwen3vl-4b --type detection  --options 5
# python answer_vqa.py --model qwen3vl-4b --type detection  --options 4


python answer_vqa.py --model qwen3vl-8b --type law  --options 5
python answer_vqa.py --model qwen3vl-8b --type law  --options 4
python answer_vqa.py --model qwen3vl-8b --type detection  --options 5
python answer_vqa.py --model qwen3vl-8b --type detection  --options 4

python answer_vqa.py --model llama --type law  --options 5
python answer_vqa.py --model llama --type law  --options 4
python answer_vqa.py --model llama--type detection  --options 5
python answer_vqa.py --model llama --type detection  --options 4

# python answer_vqa.py --model internvl-1b --type law  --options 5
# python answer_vqa.py --model internvl-1b --type law  --options 4
# python answer_vqa.py --model internvl-1b --type detection  --options 5
# python answer_vqa.py --model internvl-1b --type detection  --options 4

# python answer_vqa.py --model internvl-2b --type law  --options 5
# python answer_vqa.py --model internvl-2b --type law  --options 4
# python answer_vqa.py --model internvl-2b --type detection  --options 5
# python answer_vqa.py --model internvl-2b --type detection  --options 4


python answer_vqa.py --model internvl-8b --type law  --options 5
python answer_vqa.py --model internvl-8b --type law  --options 4
python answer_vqa.py --model internvl-8b --type detection  --options 5
python answer_vqa.py --model internvl-8b --type detection  --options 4


python answer_vqa.py --model internvl-9b --type law  --options 5
python answer_vqa.py --model internvl-9b --type law  --options 4
python answer_vqa.py --model internvl-9b --type detection  --options 5
python answer_vqa.py --model internvl-9b --type detection  --options 4


python answer_vqa.py --model internvl-14b --type law  --options 5
python answer_vqa.py --model internvl-14b --type law  --options 4
python answer_vqa.py --model internvl-14b --type detection  --options 5
python answer_vqa.py --model internvl-14b --type detection  --options 4