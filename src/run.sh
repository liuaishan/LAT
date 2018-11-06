#!/bin/bash

# vgg for cifar10 
python main.py \
	--bathsize 128 \
	--epoch 200  \
	  "/home/dsg/data/cifar10/cifar_10_pytorch/"
