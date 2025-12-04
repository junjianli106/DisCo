#!/bin/bash

cd ..
for fold in {0..3}
do
  python train.py --stage='train' --config="Cls/BRCA/MiCo/MiCo.yaml" --gpus=0 --fold=$fold
  python train.py --stage='test' --config="Cls/BRCA/MiCo/MiCo.yaml" --gpus=0 --fold=$fold
done