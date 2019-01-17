python main.py \
  --batchsize 128 \
  --epoch 30 \
  --lr 0.01 \
  --enable_lat 'False' \
  --test_flag 'False' \
  --test_data_path "/media/dsg3/dsgprivate/lat/test/new/test_data_cln.p" \
  --test_label_path "/media/dsg3/dsgprivate/lat/test/new/test_label.p" \
  --train_data_path "/media/dsg3/dsgprivate/lat/data/cifar10/" \
  --model_path "/media/dsg3/dsgprivate/yuhang/model/resnet50/lat/4step_p7e0.6a0.8/noise/" \
  --pro_num 1 \
  --alpha 0 \
  --epsilon 0 \
  --dataset "cifar10" \
  --model "resnet" \
  --dropout 'False' \
  --logfile 'log_lat_step1.txt' \
  --enable_noise 'True'
python main.py \
  --batchsize 128 \
  --epoch 20 \
  --lr 0.0005 \
  --enable_lat 'True' \
  --test_flag 'False' \
  --test_data_path "/media/dsg3/dsgprivate/lat/test/new/test_data_cln.p" \
  --test_label_path "/media/dsg3/dsgprivate/lat/test/new/test_label.p" \
  --train_data_path "/media/dsg3/dsgprivate/lat/data/cifar10/" \
  --model_path "/media/dsg3/dsgprivate/yuhang/model/resnet50/lat/4step_p7e0.6a0.8/noise/" \
  --pro_num 3 \
  --alpha 0 \
  --epsilon 1.0 \
  --dataset "cifar10" \
  --model "resnet" \
  --dropout 'True' \
  --logfile 'log_lat_step2.txt' \
  --enable_noise 'True'
python main.py \
  --batchsize 128 \
  --epoch 30 \
  --lr 0.0005 \
  --enable_lat 'True' \
  --test_flag 'False' \
  --test_data_path "/media/dsg3/dsgprivate/lat/test/new/test_data_cln.p" \
  --test_label_path "/media/dsg3/dsgprivate/lat/test/new/test_label.p" \
  --train_data_path "/media/dsg3/dsgprivate/lat/data/cifar10/" \
  --model_path "/media/dsg3/dsgprivate/yuhang/model/resnet50/lat/4step_p7e0.6a0.8/noise/" \
  --pro_num 5 \
  --alpha 0.7 \
  --epsilon 0.3 \
  --dataset "cifar10" \
  --model "resnet" \
  --dropout 'True' \
  --logfile 'log_lat_step3.txt' \
  --enable_noise 'True'
python main.py \
  --batchsize 128 \
  --epoch 30 \
  --lr 0.00005 \
  --enable_lat 'True' \
  --test_flag 'False' \
  --test_data_path "/media/dsg3/dsgprivate/lat/test/new/test_data_cln.p" \
  --test_label_path "/media/dsg3/dsgprivate/lat/test/new/test_label.p" \
  --train_data_path "/media/dsg3/dsgprivate/lat/data/cifar10/" \
  --model_path "/media/dsg3/dsgprivate/yuhang/model/resnet50/lat/4step_p7e0.6a0.8/noise/" \
  --pro_num 5 \
  --alpha 0.7 \
  --epsilon 0.3 \
  --dataset "cifar10" \
  --model "resnet" \
  --dropout 'True' \
  --logfile 'log_lat_step4.txt' \
  --enable_noise 'True'
