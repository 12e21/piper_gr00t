set -x -e

export NUM_GPUS=2

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path dataset/bi_piper_subset \
    --modality_config_path config_for_gr00tn16/bi_piper_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir outputs/bi_piper_finetune \
    --save_steps 1000 \
    --save_total_limit 5 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 64 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --use_wandb \
    --dataloader_num_workers 32 
