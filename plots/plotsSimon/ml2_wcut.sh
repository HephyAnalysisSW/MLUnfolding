#!/bin/bash
python3 ml2_wcut.py --save_model_path="/scratch-cbe/users/simon.hablas/MLUnfolding/models/EEEC_v6_older/UL2018/nAK82p-AK8pt/shortsidep1/50/all_ptcut5GeV" --plot_dir="EEEC_v6_older" --train="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v3/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/TTbar_older/ML_Data_train.npy" --val="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v3/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/TTbar_older/ML_Data_validate.npy" --training_weight_cut=1.5e-5
