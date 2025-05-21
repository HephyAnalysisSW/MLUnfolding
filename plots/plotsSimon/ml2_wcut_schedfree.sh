#!/bin/bash
#SBATCH --job-name=shsfml
#SBATCH --output=/scratch/simon.hablas/batch_output/job_%j_%A_%a.out
#SBATCH --error=/scratch/simon.hablas/batch_output/job_%j_%A_%a.err
#SBATCH --qos=medium
#SBATCH --time=1-18:00:00
#SBATCH --partition=c
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --array=6

echo "Job started on $(hostname) at $(date)"

# Load Conda
module load miniconda3/23.9.0-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sf_env

conda info --envs

echo "Python path:"
which python
python -c "import sys; print('Python executable:', sys.executable)"


# Define parameter grid
NODES=(64 128 256)
LAYERS=(3 6 9)
DEPTHS=(1 2)

# Generate combinations
COMBOS=()
for n in "${NODES[@]}"; do
  for l in "${LAYERS[@]}"; do
    for d in "${DEPTHS[@]}"; do
      COMBOS+=("$n $l $d")
    done
  done
done

# Select based on job array index
PARAMS=(${COMBOS[$SLURM_ARRAY_TASK_ID]})
N=${PARAMS[0]}
L=${PARAMS[1]}
D=${PARAMS[2]}



VERSION=$((90 + SLURM_ARRAY_TASK_ID))
echo "VERSION=$VERSION"

echo "Model v{VERSION}: CiNN | 200 Epochen | Schedfree | v4 combined_Data | nodes=${N}, layers=${L}, depth=${D} |"


~/.conda/envs/sf_env/bin/python -u ml2_wcut_schedfree.py \
--save_model_path="/scratch-cbe/users/simon.hablas/MLUnfolding/models/EEEC_v${VERSION}_older/UL2018_batch256/nAK82p-AK8pt/shortsidep1/50/all_ptcut5GeV/" \
--train="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/ML_Data_train_combined.npy" \
--val="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/ML_Data_validate_combined.npy" \
--training_weight_cut=1.5e-5 \
--nodes=$N \
--layers=$L \
--networkdepth=$D
