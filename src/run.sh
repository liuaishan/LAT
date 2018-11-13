#!/bin/bash
python main.py \
  --batchsize 128 \
  --epoch 100 \
  --lr 0.0005 \
  --enable_lat 'True' \
  --test_flag 'False' \
  --test_data_path "/media/dsg3/dsgprivate/lat/test/test_data_cln.p" \
  --test_label_path "/media/dsg3/dsgprivate/lat/test/test_label.p" \
  --train_data_path "/media/dsg3/dsgprivate/lat/data/cifar10/" \
  --model_path "/media/dsg3/dsgprivate/lat/model/vgg/" \
  --pro_num 3 \
  --alpha 0.3 \
  --epsilon 0.3 \
  --dataset "cifar10" \
  --model "vgg" 
