# BAM: Unsupervised Representation Learning by Balanced Self Attention Matching‏ (ECCV 2024)

This repository provides the official PyTorch implementation for **BAM** (**B**alanced Self-**A**ttention **M**atching),
as described in the paper [Unsupervised Representation Learning by Balanced Self Attention Matching](https://arxiv.org/abs/2408.02014) (Accepted by ECCV 2024).

Many leading self-supervised methods for unsupervised representation learning, in particular those for embedding image features, are built on variants of the instance discrimination task, whose optimization is known to be prone to instabilities that can lead to feature collapse. Different techniques have been devised to circumvent this issue, including the use of negative pairs with different contrastive losses, the use of external memory banks, and breaking of symmetry by using separate encoding networks with possibly different structures. Our method, termed BAM, rather than directly matching features of different views (augmentations) of input images, is based on matching their self-attention vectors, which are the distributions of similarities to the entire set of augmented images of a batch.

## Pretrained models

You can choose to download only the weights of the pretrained backbone used for downstream tasks, or the full checkpoint which contains backbone and projection head weights for both student and teacher networks.

| arch          | params                 | linear        | download      |
| ------------- |-------------           | ------------- | ------------- |
| ViT-B/16      | 85M                    | 78.1          | --            |
| CAFormer-M36  | 57M                    | 78.9          | --            |

## Training

We provide the script for pretraining our BAM models under different configurations. All scripts can be found under the "scripts" directory.

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

