python main.py \
  --batchsize 200 \
  --epoch 80 \
  --lr 0.00005 \
  --enable_lat 'True' \
  --test_flag 'True' \
  --adv_flag 'True' \
  --train_data_path "/media/dsg3/dsgprivate/lat/data/sampled_imagenet/" \
  --model_path "/media/dsg3/dsgprivate/yuhang/model/alexnet/newnewnewplat/" \
  --pro_num 5 \
  --alpha 0.7 \
  --epsilon 0.06 \
  --dataset "imagenet" \
  --model "alexnetBN" \
  --dropout 'True' \
  --test_data_path "/media/dsg3/dsgprivate/lat/sampled_imagenet/test/alexnet/test_data_cln.p" \
  --test_label_path "/media/dsg3/dsgprivate/lat/sampled_imagenet/test/alexnet/test_label.p" \
  --logfile 'log8.txt'
