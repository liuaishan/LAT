#!/bin/bash
python main.py \
  --batchsize 128 \
  --epoch 80 \
  --lr 0.001 \
  --enable_lat 'True' \
  --test_flag 'True' \
  --test_data_path "/home/dsg/yuhang/src/test/test_data_cln.p" \
  --test_label_path "/home/dsg/yuhang/src/test/test_label.p" \
  --train_data_path "/home/dsg/data/cifar10/cifar_10_pytorch/" \
  --model_path "/home/dsg/yuhang/src/model/" \
  --pro_num 2 \
  --alpha 0.5 \
  --epsilon 0.1 \
  --dataset "cifar10" \
  --model "resnet" 
