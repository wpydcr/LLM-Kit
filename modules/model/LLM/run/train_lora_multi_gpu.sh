# 同一服务器同时运行多个训练代码，需要手动指定main_process_port，避免冲突
# int8不可以和多卡同时使用
nohup accelerate launch \
    --config_file ../accelerate_configs/multi_gpu.yaml \
    --main_process_port 29501 \
    ../train.py \
        --use_lora \
        --gradient_checkpoint \
> nohup_lora_multi_gpu.out &
