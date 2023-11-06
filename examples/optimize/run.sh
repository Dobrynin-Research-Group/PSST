#!/usr/bin/bash

cwd=$(pwd)
output_base_dir=${cwd}/../train

output_filename=adam_config.yaml
generator_file=${cwd}/../../generator_configs/AridAgar.yaml
optimizer_file=${cwd}/optimizer_config.yaml

for model in "Inception3" "Vgg13"; do
    mkdir ${model}
    cd ${model}
    for param in "bg" "bth"; do
        mkdir ${param}
        cd ${param}

        outdir=${output_base_dir}/${model}/${param}
        mkdir -p ${outdir}
        outfile=${outdir}/${output_filename}

        python3 ../../optimize.py ${outfile} -p ${param} -m ${model} -g ${generator_file} -c ${optimizer_file}

        cd ..
    done
    cd ..
done
