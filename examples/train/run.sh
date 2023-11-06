#!/usr/bin/bash

cwd=$(pwd)

checkpoint_file=chkpt.pt
generator_file=${cwd}/../../generator_configs/AridAgar.yaml
adam_file=adam_config.yaml
run_file=${cwd}/run_config.yaml

for model in "Inception3" "Vgg13"; do
    cd ${model}
    for param in "bg" "bth"; do
        cd ${param}
        python3 ../../optimize.py ${checkpoint_file} -p ${param} -m ${model} \
            -g ${generator_file} -a ${adam_config} -c ${run_file}
        cd ..
    done
    cd ..
done
