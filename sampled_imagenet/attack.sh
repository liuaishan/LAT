python attack.py \
  --attack 'ifgsm' \
  --generate 'True' \
  --droplast 'False' \
  --model 'alexnet' \
  --enable_lat 'False'\
  --modelpath "/media/dsg3/dsgprivate/zhangchongzhi/model/alex_naive/alex_naive.pkl" \
  --model_batchsize 100 \
  --dropout 'True' \
  --dataset 'imagenet' \
  --attack_batchsize 64 \
  --attack_epsilon 8 \
  --attack_alpha 1 \
  --attack_iter 10 \
  --attack_momentum 1.0 \
  --savepath "/media/dsg3/dsgprivate/lat/sampled_imagenet/test_momentum/alexnet/" \
  --lat_epsilon 0.6 \
  --lat_pronum 7 \
  
