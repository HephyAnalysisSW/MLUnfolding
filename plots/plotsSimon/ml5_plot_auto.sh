#!/bin/bash
#SBATCH --job-name=shmlplt
#SBATCH --output=/scratch/simon.hablas/batch_output/job_%j_%A_%a.out
#SBATCH --error=/scratch/simon.hablas/batch_output/job_%j_%A_%a.err
#SBATCH --qos=medium
#SBATCH --time=1-12:00:00
#SBATCH --partition=c
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=6G
#SBATCH --array=15

echo "Job started on $(hostname) at $(date)"

# Load Conda
module load miniconda3/23.9.0-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sf_env

conda info --envs

echo "Python path:"
which python
python -c "import sys; print('Python executable:', sys.executable)"


VERSION=$((70 + SLURM_ARRAY_TASK_ID))
echo "VERSION=$VERSION"

~/.conda/envs/sf_env/bin/python -u ml5_plot.py --load_model_path="/scratch-cbe/users/simon.hablas/MLUnfolding/models/EEEC_v${VERSION}_older/UL2018/nAK82p-AK8pt/shortsidep1/50/all_ptcut5GeV/weight_cut_1p5e-05" --plot_dir="/groups/hephy/cms/simon.hablas/www/MLUnfolding/plots/EEEC_${VERSION}_old_data" --train="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/ML_Data_train_combined.npy"  --val="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/TTbar_older/ML_Data_validate.npy"  --info="Sample Data : mt= 172.5 GeV" --weight_cut=1.5e-05