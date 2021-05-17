# HowTo100M code

This repo provides code from the HowTo100M paper.
We provide implementation of:
- Our training procedure on HowTo100M for learning a joint text-video embedding
- Our evaluation code on MSR-VTT, YouCook2 and LSMDC for Text-to-Video retrieval
- A pretrain model on HowTo100M
- Feature extraction from raw videos script we used

More information about HowTo100M can be found on the project webpage: https://www.di.ens.fr/willow/research/howto100m/


# Requirements
- Python 3
- PyTorch (>= 1.0)
- gensim


## Video feature extraction

This separate github repo: https://github.com/antoine77340/video_feature_extractor
provides an easy to use script to extract the exact same 2D and 3D CNN features we have extracted in our work.

## Downloading a pretrained model
This will download our pretrained text-video embedding model on HowTo100M.

```
mkdir model
cd model
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/models/howto100m_pt_model.pth
cd ..
```

## Downloading meta-data for evaluation (csv, pre-extracted features for evaluation, word2vec)
This will download all the data needed for evaluation.

```
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/metadata_eval.zip
unzip -d metadata_eval.zip
```

## Example scripts
```
CUDA_VISIBLE_DEVICES=7 python train.py --epic 1 --eval_epic 1 --pretrain_path=model/howto100m_pt_model.pth --clips_per_sample 2 --lr 0.0001 --clips_before 1
CUDA_VISIBLE_DEVICES=7 python train.py --epic 1 --eval_epic 1 --pretrain_path=model/howto100m_pt_model.pth --clips_per_sample 3 --lr 0.0001 --clips_before 1
CUDA_VISIBLE_DEVICES=6 python train.py --epic 1 --eval_epic 1 --pretrain_path=model/howto100m_pt_model.pth --clips_per_sample 2 --lr 0.0001 --epic_gt_verb 1 --epochs 100
CUDA_VISIBLE_DEVICES=6 python train.py --epic 1 --eval_epic 1 --pretrain_path=model/howto100m_pt_model.pth --clips_per_sample 3 --lr 0.0001 --epic_gt_verb 1 --epochs 100
```

## If you find the code / model useful, please cite our paper
```
@inproceedings{miech19howto100m,
   title={How{T}o100{M}: {L}earning a {T}ext-{V}ideo {E}mbedding by {W}atching {H}undred {M}illion {N}arrated {V}ideo {C}lips},
   author={Miech, Antoine and Zhukov, Dimitri and Alayrac, Jean-Baptiste and Tapaswi, Makarand and Laptev, Ivan and Sivic, Josef},
   booktitle={ICCV},
   year={2019},
}
```
