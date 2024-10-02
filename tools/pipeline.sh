#!/bin/bash

gcard=$1
lund=$2
gemc_hipo=$3
cooked_hipo=$4
dst_hipo=$5
train_csv=$6
train_h5=$7
yaml_config=$8

#gemc ${gcard} -SAVE_ALL_MOTHERS=1 -SKIPREJECTEDHITS=1 -NGENP=50 -INTEGRATEDRAW="*" -USE_GUI=0 -RUNNO=11 -INPUT_GEN_FILE="LUND, ${lund}" -OUTPUT="hipo, ${gemc_hipo}"

echo "******************************************************************************************"
echo "** Finished running GEMC --> Generated Hipo File at ${gemc_hipo}"
echo "******************************************************************************************"

#/work/clas12/users/gmat/clas12/coatjava_dev/coatjava/coatjava/bin/recon-util -i ${gemc_hipo} -o ${cooked_hipo} -y ${yaml_config}

echo "******************************************************************************************"
echo "** Finished running RECON-UTIL --> Generated Hipo File at ${cooked_hipo}"
echo "******************************************************************************************"

FILTER_BANKS='RUN::*,MC::*,REC::Particle,REC::Calorimeter,REC::Track,REC::Traj,ECAL::*'
#hipo-utils -filter -b ${FILTER_BANKS} -merge -o ${dst_hipo} ${cooked_hipo}

echo "******************************************************************************************"
echo "** Finished running HIPO-UTILS -FILTER --> Generated Hipo File at ${dst_hipo}"
echo "******************************************************************************************"

python3 tools/process_ecal_data_csv.py ${dst_hipo} ${train_csv} "mc"

echo "************************************************************************************************"
echo "** Finished running PYTHON3 tools/process_ecal_data_csv.py --> Generated CSV File at ${train_csv}"
echo "************************************************************************************************"

python3 tools/process_ecal_data_tensors.py ${train_csv} ${train_h5}

echo "*****************************************************************************************************"
echo "** Finished running PYTHON3 tools/process_ecal_data_tensors.py --> Generated CSV File at ${train_h5}"
echo "*****************************************************************************************************"



