#!/bin/bash
python main.py \
  --batchsize 128 \
  --epoch 80 \
  --lr 0.001 \
  --enable_lat 'True' \
  --test_flag 'False' \
  --test_data_path "/home/dsg/yuhang/src/test/test_adv(eps_0.2).p" \
  --test_label_path "/home/dsg/yuhang/src/test/test_label.p" \
  --train_data_path "/home/dsg/data/cifar10/cifar_10_pytorch/" \
  --model_path "/home/dsg/yuhang/src/model/" \
  --pro_num 1 \
  --alpha 0.01 \
  --epsilon 0.1 \
  --dataset "cifar10" \
  --model "resnet" 
