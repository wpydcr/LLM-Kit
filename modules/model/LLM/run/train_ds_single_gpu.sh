# 建议150G以上内存
nohup accelerate launch \
    --config_file ../accelerate_configs/ds_single_gpu.yaml \
    ../train.py \
        --gradient_checkpoint \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
> nohup_ds_single_gpu.out &
