nohup accelerate launch \
    --config_file ../accelerate_configs/multi_gpu.yaml \
    ../train.py \
        --gradient_checkpoint \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
> nohup_multi_gpu.out &
