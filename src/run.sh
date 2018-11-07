#!/bin/bash
python main.py \
  --batchsize 128 \
  --epoch 80 \
  --lr 0.005 \
  --enable_lat 'True' \
  --test_flag 'False' \
  --test_data_path "/home/dsg/yuhang/src/test/" \
  --train_data_path "/home/dsg/data/cifar10/cifar_10_pytorch/" \
  --model_path "/home/dsg/yuhang/src/model/" \
  --pro_num 10 \
  --alpha 0.8 \
  --epsilon 0.3 \
  --dataset "cifar10" \
  --model "resnet" 
