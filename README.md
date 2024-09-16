# BAM: Unsupervised Representation Learning by Balanced Self Attention Matching‏ (ECCV 2024)

This repository provides the official PyTorch implementation for **BAM** (**B**alanced Self-**A**ttention **M**atching),
as described in the paper [Unsupervised Representation Learning by Balanced Self Attention Matching](https://arxiv.org/abs/2408.02014) (Accepted by ECCV 2024).

Many leading self-supervised methods for unsupervised representation learning, in particular those for embedding image features, are built on variants of the instance discrimination task, whose optimization is known to be prone to instabilities that can lead to feature collapse. Different techniques have been devised to circumvent this issue, including the use of negative pairs with different contrastive losses, the use of external memory banks, and breaking of symmetry by using separate encoding networks with possibly different structures. Our method, termed BAM, rather than directly matching features of different views (augmentations) of input images, is based on matching their self-attention vectors, which are the distributions of similarities to the entire set of augmented images of a batch.

## Pretrained models

We provide the pretrained weights for some of our best configurations -

| arch          | params                 | linear        | download      |
| ------------- |-------------           | ------------- | ------------- |
| ViT-B/16      | 85M                    | 78.1          | --            |
| CAFormer-M36  | 57M                    | 78.9          | --            |

## Training

We provide the script for pretraining our BAM models under different configurations. All scripts can be found under the "scripts" directory.

### BAM specific hyperparameters -

    --reg 0.05                   # Entropy regularization for the Sinkhorn algorithm (used for target pairwise attention). smaller value -> smaller entropy.
    --temperature 0.1            # Temperature for the Softmax attention. smaller value -> smaller entropy.
    --num_sink_iter 3            # Number of Sinkhorn iterations. 
    --top_k_sink 0               # We found that using only the top-k values (e.g. 128) of the target attention can slightly improve linear accuracy. This parameter does not mentioned in the paper.
    --positive_masking True      # Use positive masking. We usually set this to True.
    --target_crops_number 2      # Use this amount of views for creating the target attention matrix. We use the same number of large crops.

## Online evaluation

We monitor BAM pretraining using a K-NN classifier. For faster evaluation, we use 10% of imagenet training data. The implementation support distributed machines.
You can define the evaluation configuration using the following arguments -

    --knn_freq 20                # evaluate every 20 epochs 
    --knn_train_fraction 0.1     # use 10% of training data 
    --knn_eval_fraction 1.       # use 100% of validation data

## Linear probing

Scripts for running linear head on top of forzen BAM backbone can be found under the "scripts" directory.

## Citation

<p>

#### If you find this repository useful in your research, please cite:
    @article{shalam2024unsupervised,
      title={Unsupervised Representation Learning by Balanced Self Attention Matching},
      author={Shalam, Daniel and Korman, Simon},
      journal={arXiv preprint arXiv:2408.02014},
      year={2024}
    }
    
</p>

## Acknowledgment
[Emerging Properties in Self-Supervised Vision Transformers](https://github.com/facebookresearch/dino)

