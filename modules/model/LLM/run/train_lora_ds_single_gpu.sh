nohup accelerate launch \
    --config_file ../accelerate_configs/ds_single_gpu.yaml \
    ../train.py \
        --use_lora \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_checkpoint \
> nohup_lora_ds_single_gpu.out &
