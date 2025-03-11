# Pretrain on VidCab
## Generative Retrieval (Ours)
```bash
python train.py \
    --exp_id='vidcab_ret_lastv_m' \
    --dataset_dir='/blob/v-lqinghong/data/Ego_database' \
    --log_base_dir='/blob/v-lqinghong/experiments/VLog' \
    --llm_model='gpt2-medium' \
    --vis_model='openai/clip-vit-base-patch32' \
    --dataset='ret' \
    --val_dataset='ret' \
    --metadata='egoclip_vidcab' \
    --val_metadata='egoclip_vidcab' \
    --epochs=10 \
    --batch_size=32 \
    --val_batch_size=32 \
    --print_freq=1 \
    --precision='bf16' \
    --hidden_dim=1152 \
    --workers=16 \
    --max_len=128 \
    --max_len_eval=128 \
    --max_clip_len=128 \
    --vocab_model 'google/siglip-so400m-patch14-384' \
    --last_vis_mean
```

## Generative Modeling
If you want to run on the default language modeling (e.g. cross entropy loss on the output tokens).
```bash
python train.py \
    --exp_id='vidcab_gen_m' \
    --dataset_dir='/blob/v-lqinghong/data/Ego_database' \
    --log_base_dir='/blob/v-lqinghong/experiments/VLog' \
    --llm_model='gpt2-medium' \
    --vis_model='openai/clip-vit-base-patch32' \
    --dataset='gen' \
    --val_dataset='gen' \
    --metadata='egoclip_vidcab' \
    --val_metadata='egoclip_vidcab' \
    --epochs=10 \
    --batch_size=32 \
    --val_batch_size=32 \
    --print_freq=1 \
    --precision='bf16' \
    --hidden_dim=1152 \
    --workers=16 \
    --max_len=128 \
    --max_len_eval=128 \
    --max_clip_len=128 \
    --vocab_model 'google/siglip-so400m-patch14-384'
```

# COIN
Replace the `metadata` and `val_metadata` to `coin_step`, `coin_next` and `coin_task` to run them individually.

```bash
python train.py \
    --exp_id='coin-step'  \
    --dataset_dir='/blob/v-lqinghong/data/Ego_database'  \
    --log_base_dir='/blob/v-lqinghong/experiments/VLog'  \
    --llm_model='gpt2-medium'  \
    --vis_model='openai/clip-vit-base-patch32'  \
    --dataset='coin'  \
    --val_dataset='coin'  \
    --metadata='coin_step'  \
    --val_metadata='coin_step'  \
    --epochs=100  \
    --batch_size=24  \
    --val_batch_size=24  \
    --print_freq=1  \
    --precision='bf16'  \
    --hidden_dim=1152  \
    --workers=16  \
    --num_frame=8  \
    --max_len=128  \
    --max_len_eval=128  \
    --visual_input='feature'  \
    --vocab_model='google/siglip-so400m-patch14-384'  \
    --train_class  \
    --last_vis_mean
```

# EgoSchema