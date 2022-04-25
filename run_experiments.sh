#!/bin/bash

params="1 10 100 1000"
arr=($params)

for ((i=0; i<=3; i++))
do
    # echo train_fast_isd_output/lambda${arr[i]}
    CUDA_VISIBLE_DEVICES="$i" python fast_isd.py  --gama "${arr[i]}" --out-dir "train_fast_isd_output/gama${arr[i]}" --epochs 15 &
done

wait

