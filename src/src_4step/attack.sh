python attack.py \
  --attack 'momentum_ifgsm' \
  --generate 'True' \
  --droplast 'False' \
  --model 'resnet' \
  --enable_lat 'False'\
  --modelpath "/media/dsg3/dsgprivate/yuhang/model/resnet50/naive/naive_param.pkl" \
  --model_batchsize 128 \
  --dropout 'False' \
  --dataset 'cifar10' \
  --attack_batchsize 128 \
  --attack_epsilon 8 \
  --attack_alpha 1 \
  --attack_iter 10 \
  --attack_momentum 1.0 \
  --savepath "/media/dsg3/dsgprivate/lat/test_momentum/resnet/" \
  --lat_epsilon 0.6 \
  --lat_pronum 7 \
  
