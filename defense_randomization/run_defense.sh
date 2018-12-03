python defense.py \
  --batch_size 128 \
  --img_size 32 \
  --img_resize 35 \
  --iter_times 10 \
  --num_classes 10 \
  --test_data_path "/media/dsg3/dsgprivate/lat/test/new/test_data_cln.p" \
  --test_label_path "/media/dsg3/dsgprivate/lat/test/new/test_label.p" \
  --model 'vgg' \
  --model_path "/media/dsg3/dsgprivate/yuhang/model/adv_training/dropout/naive_param.pkl"
