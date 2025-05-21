#!/bin/bash
python3 ml_plot_input_compare.py \
--file1="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/25/ptcut0GeV/TTbar_older/ML_Data_train.npy" --l1="25 best >=0Gev" \
--file2="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/25/ptcut5GeV/TTbar_older/ML_Data_train.npy" --l2="25 best >=5Gev" \
--file3="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/25/ptcut10GeV/TTbar_older/ML_Data_train.npy" --l3="25 best >=10Gev"

python3 ml_plot_input_compare.py \
--file1="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut0GeV/TTbar_older/ML_Data_train.npy" --l1="50 best >=0Gev" \
--file2="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/TTbar_older/ML_Data_train.npy" --l2="50 best >=5Gev" \
--file3="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut10GeV/TTbar_older/ML_Data_train.npy" --l3="50 best >=10Gev"

python3 ml_plot_input_compare.py \
--file1="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/10/ptcut0GeV/TTbar_older/ML_Data_train.npy" --l1="10 best >=0Gev" \
--file2="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/10/ptcut5GeV/TTbar_older/ML_Data_train.npy" --l2="10 best >=5Gev" \
--file3="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/10/ptcut10GeV/TTbar_older/ML_Data_train.npy" --l3="10 best >=10Gev"

python3 ml_plot_input_compare.py \
--file1="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/100/ptcut0GeV/TTbar_older/ML_Data_train.npy" --l1="100 best >=0Gev" \
--file2="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/100/ptcut5GeV/TTbar_older/ML_Data_train.npy" --l2="100 best >=5Gev" \
--file3="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/100/ptcut10GeV/TTbar_older/ML_Data_train.npy" --l3="100 best >=10Gev"

python3 ml_plot_input_compare.py \
--file1="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/10/ptcut5GeV/TTbar_older/ML_Data_train.npy" --l1="10 best >=10Gev" \
--file2="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/25/ptcut5GeV/TTbar_older/ML_Data_train.npy" --l1="25 best >=25Gev" \
--file3="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/50/ptcut5GeV/TTbar_older/ML_Data_train.npy" --l2="50 best >=50Gev" \
--file4="/scratch-cbe/users/simon.hablas/MLUnfolding/data/EEEC_v4/UL2018/nAK82p-AK8pt/shortsidep1/100/ptcut5GeV/TTbar_older/ML_Data_train.npy" --l3="100 best >=100Gev"