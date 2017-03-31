#!/bin/bash

set -x

#$ -cwd

# ask for 2 CPUs
#$ -pe smp 2

# long queue
#$ -q long.q

# more memory
## #$ -l h_vmem=5G


# Anaconda:
#export PATH=/swshare/anaconda/bin:$PATH
# echo Which anaconda:
# which python

#source /afs/cern.ch/sw/lcg/external/gcc/4.9/x86_64-slc6-gcc49-opt/setup.sh
#source /afs/cern.ch/sw/lcg/app/releases/ROOT/6.06.08/x86_64-slc6-gcc49-opt/root/bin/thisroot.sh


# added by Janik in order to give python the path to xgboost
#export PYTHONPATH=~/xgboost/python-package:${PYTHONPATH}
#export PYTHONPATH=/afs/cern.ch/sw/lcg/app/releases/ROOT/6.06.08/x86_64-slc6-gcc49-opt/root/lib:${PYTHONPATH}
# echo "xgboost and ROOT path exported"

source $HOME/.bashrc



[[ -z $JOB_ID ]] && JOB_ID=$$

notebook=$1 && shift
## config=$1 && shift
output_dir=$1 && shift


input_dir=$(dirname $notebook)
[[ "$input_dir" != /* ]] && [[ "$input_dir" != *:/* ]] && input_dir=$PWD/$input_dir

[[ "$config" != /* ]] && [[ "$config" != *:/* ]] && config=$PWD/$config

notebook=$(basename $notebook | sed 's%.ipynb$%%')

output=${notebook}_$(basename $config | sed 's%.json$%%')_${JOB_ID}


## set +x
## source $HOME/.sge_env.sh
## set -x

export PYTHONPATH=${input_dir}:$PYTHONPATH


scratch=/scratch/$USER
if [[ ! -d $scratch ]]; then
    mkdir $scratch || exit -1
fi


workdir=$scratch/job-${JOB_ID}
if [[ ! -d $workdir ]]; then 
    mkdir $workdir || exit -1 
fi

output_dir=${output_dir}/${output}
if [[ ! -d $output_dir ]]; then 
    mkdir $output_dir || exit -1 
fi

cd $workdir

cp -p $input_dir/${notebook}.ipynb ${output}.ipynb

cat >> io.json <<EOF
{
    "inputDir" : "${input_dir}",
    "outName" : "${output}",
    "outDir" : "."
}
EOF

## my_train_config=$config,io.json jupyter nbconvert --allow-errors --debug --execute --to notebook --ExecutePreprocessor.timeout=-1  --inplace $output
my_train_config=io.json jupyter nbconvert --allow-errors --debug --execute --to notebook --ExecutePreprocessor.timeout=-1  --inplace $output

echo ${ouptut}
find . -name \*

for file in ${output}*.{ipynb,pkl.gz,root}; do
    cp -p $file ${output_dir} || echo "Failed to copy $file"
done

cd $scratch

rm -rf $workdir
