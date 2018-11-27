CUDA_VISIBLE_DEVICES=4 python attack.py \
  --attack 'fgsm' \
  --generate 'False' \
  --droplast 'False' \
  --model 'vgg' \
  --enable_lat 'False'\
  --modelpath "/media/dsg3/dsgprivate/lat/liuaishan/cifar10/vgg16_origin_dropout/naive_param.pkl" \
  --model_batchsize 128 \
  --dropout 'True' \
  --dataset 'cifar10' \
  --attack_batchsize 128 \
  --attack_epsilon 8.0 \
  --attack_alpha 2.0 \
  --attack_iter 10 \
  --savepath "/media/dsg3/dsgprivate/lat/test" \
  --lat_epsilon 0.3 \
  --lat_pronum 5 \
  