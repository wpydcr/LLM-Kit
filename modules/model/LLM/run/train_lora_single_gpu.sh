# 同一服务器同时运行多个训练代码，需要手动指定main_process_port，避免冲突
nohup accelerate launch \
    --config_file ../accelerate_configs/single_gpu.yaml \
    --main_process_port 29501 \
    ../train.py \
        --use_lora \
        --use_8bit \
        --gradient_checkpoint \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
> nohup_lora_single_gpu.out &
