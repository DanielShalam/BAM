#!/bin/bash

#SBATCH -J Liner-ViT-B
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH -D <set_output_dir>
#SBATCH --ntasks=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --exclusive


### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=54341
export WORLD_SIZE=8


echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


srun --container-image=/users/korman/dshalam/container.sqsh /bin/bash -c  "python -u /root/contrastive_mixup/eval_linear.py \
	--use_teacher true \
    --wandb true \
    --seed 1 \
	--use_fp16 false \
    --arch vit_base \
    --num_workers 8 \
    --epochs 100 \
    --lr 0.002 \
    --batch_size_per_gpu 128 \
    --bn_head true \
    --n_last_blocks 1 \
    --avgpool_patchtokens true \
    --val_freq 1 \
    --data_path /your_imagenet_path/ \
    --pretrained_weights /your_checkpoint_dir/checkpoint.pth \
    --output_dir /your_output_dir/ "

