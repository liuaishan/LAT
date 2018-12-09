python attack.py \
  --attack 'bpda' \
  --generate 'False' \
  --droplast 'True' \
  --model 'vgg' \
  --enable_lat 'False'\
  --modelpath "/media/dsg3/dsgprivate/lat/liuaishan/cifar10/vgg16_origin_dropout/naive_param.pkl" \
  --model_batchsize 128 \
  --dropout 'True' \
  --dataset 'cifar10' \
  --attack_batchsize 128 \
  --attack_epsilon 0.03 \
  --attack_alpha 0.005 \
  --attack_iter 6 \
  --savepath "/media/dsg3/dsgprivate/lat/test/densenet/" \
  --lat_epsilon 0.6 \
  --lat_pronum 7 \
  
