#!/bin/bash


#SBATCH -J BAM-Pretrain
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH -D <set_output_dir>
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --nodes=2
#SBATCH --exclusive


export MASTER_PORT=53334
export WORLD_SIZE=16


### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


srun --container-image=<your_container_path> /bin/bash -c  "python -u /root/BAM/main_pretrain.py \
	--grad_checkpointing 4 \
    --wandb true \
    --seed 1 \
	--teacher true \
    --num_workers 10 \
    --arch caformer_m36 \
    --epochs 400 \
	--warmup_epochs 40 \
    --batch_size_per_gpu 256 \
	--loc_interval 40 \
    --knn_freq 16 \
    --n_layers 3 \
    --hidden_dim 4096 \
    --out_dim 4096 \
	--proj_use_bn false \
    --proj_last_bn false \
	--proj_act gelu \
	--proj_bias true \
    --optimizer adamw \
    --drop_path_rate 0. \
	--qkv_bias true \
	--reg 0.05 \
	--reg_final 0.05 \
	--temperature 0.1 \
	--num_sink_iter 3 \
    --lr 0.0003 \
	--min_lr 1e-3 \
    --clip_grad 0.3 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
	--target_crops_number 2 \
	--global_crops_number 2 \
    --global_crops_scale 0.25 1. \
	--local_crops_number 6 \
	--local_crops_scale 0.05 0.25 \
    --use_fp16 true \
	--top_k_sink 128 \
    --data_path /your_imagenet_path/train \
	--output_dir /your_output_dir_path/ "

