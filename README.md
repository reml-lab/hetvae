
# Heteroscedastic Temporal Variational Autoencoder for Irregularly Sampled Time Series (HetVAE)

This repository is an official implementation of Heteroscedastic Temporal Variational Autoencoder for Irregularly Sampled Time Series.
HeTVAE is a deep learning framework for probabilistic interpolation of irregularly sampled or sparse time series data.


## Requirements

To run this project, you will need to install the requirements. 
The code requires Python 3.7 or later. The file [requirements.txt](requirements.txt) contains the full list of required Python modules.

```bash
pip3 install -r requirements.txt
```


## Training and Evaluation

To reproduce the results in the paper, run the following commands:

1. PhysioNet Dataset
```bash
python3 train.py --niters 2000 --lr 0.0001 --batch-size 128 --rec-hidden 128 --latent-dim 128 --width 128 --embed-time 128 --enc-num-heads 1 --num-ref-points 16 --n 8000 --dataset physionet --seed 1 --save --norm --intensity --net hetvae --bound-variance --shuffle  --sample-tp 0.5 --elbo-weight 1.0 --mse-weight 5.0 --mixing concat --k-iwae 1
```
2. MIMIC-III Dataset
```bash
python3 train.py --niters 2000 --lr 0.0001 --batch-size 128 --rec-hidden 128 --latent-dim 128 --width 512 --embed-time 128 --enc-num-heads 1 --num-ref-points 16 --dataset mimiciii --seed 1 --save --norm --intensity --net hetvae --bound-variance --shuffle  --sample-tp 0.5 --elbo-weight 1.0 --mse-weight 5.0 --mixing concat --k-iwae 1
```
3. Synthetic Dataset
```bash
python3 train.py --niters 2000 --lr 0.0001 --batch-size 128 --rec-hidden 16 --latent-dim 64 --width 512 --embed-time 128 --enc-num-heads 1 --num-ref-points 16 --n 2000 --dataset toy --seed 0 --save --norm --intensity --net hetvae --bound-variance --shuffle  --sample-tp 0.5 --elbo-weight 1.0 --mse-weight 1.0 --mixing concat --k-iwae 1
```


## Demo

This [notebook](src/synthetic_data_interpolation.ipynb) provides an example to reproduce the visualizations in the paper on synthetic dataset.


## Ablations

Different components of the HeTVAE model are denoted as: HET: heteroscedastic output layer, ALO: augmented learning objective, INT: intensity encoding, DET: deterministic pathway. To reproduce the ablation results on PhysioNet, run the following commands:

1. HeTVAE - ALO
```bash
python3 train.py --niters 2000 --lr 0.0001 --batch-size 128 --rec-hidden 32 --latent-dim 32 --width 128 --embed-time 128 --enc-num-heads 1 --num-ref-points 8 --n 8000 --dataset physionet --seed 1 --save --norm --intensity --net hetvae --bound-variance --shuffle  --sample-tp 0.5 --elbo-weight 1.0 --mse-weight 0.0 --mixing concat --k-iwae 1
```
2. HeTVAE - DET
```bash
python3 train.py --niters 2000 --lr 0.0001 --batch-size 128 --rec-hidden 128 --latent-dim 64 --width 512 --embed-time 128 --enc-num-heads 1 --num-ref-points 8 --n 8000 --dataset physionet --seed 1 --save --norm --intensity --net hetvae_det --bound-variance --shuffle  --sample-tp 0.5 --elbo-weight 1.0 --mse-weight 10.0 --mixing concat --k-iwae 1
```
3. HeTVAE - INT
```bash
python3 train.py --niters 2000 --lr 0.0001 --batch-size 128 --rec-hidden 64 --latent-dim 128 --width 128 --embed-time 128 --enc-num-heads 1 --num-ref-points 8 --n 8000 --dataset physionet --seed 1 --save --norm --intensity --net hetvae --bound-variance --shuffle  --sample-tp 0.5 --elbo-weight 1.0 --mse-weight 5.0 --mixing interp_only --k-iwae 1
```
4. HeTVAE - HET - ALO 
```bash
python3 train.py --niters 2000 --lr 0.0001 --batch-size 128 --rec-hidden 128 --latent-dim 128 --width 512 --embed-time 128 --enc-num-heads 1 --num-ref-points 8 --n 8000 --dataset physionet --seed 1 --save --norm --intensity --net hetvae --shuffle  --sample-tp 0.5 --elbo-weight 1.0 --mse-weight 0.0 --mixing concat --k-iwae 1 --const-var --std 0.8
```
5. HeTVAE - DET - ALO
```bash
python3 train.py --niters 2000 --lr 0.0001 --batch-size 128 --rec-hidden 128 --latent-dim 32 --width 256 --embed-time 128 --enc-num-heads 1 --num-ref-points 8 --n 8000 --dataset physionet --seed 1 --save --norm --intensity --net hetvae_det --bound-variance --shuffle  --sample-tp 0.5 --elbo-weight 1.0 --mse-weight 0.0 --mixing concat --k-iwae 1
```
6. HeTVAE - PROB - ALO
```bash
python3 train.py --niters 2000 --lr 0.0001 --batch-size 128 --rec-hidden 32 --latent-dim 128 --width 128 --embed-time 128 --enc-num-heads 1 --num-ref-points 8 --n 8000 --dataset physionet --seed 1 --save --norm --intensity --net hetvae_prob --bound-variance --shuffle  --sample-tp 0.5 --elbo-weight 1.0 --mse-weight 0.0 --mixing concat --k-iwae 1 --kl-zero
```
7. HeTVAE - INT - DET - ALO
```bash
python3 train.py --niters 2000 --lr 0.0001 --batch-size 128 --rec-hidden 128 --latent-dim 64 --width 512 --embed-time 128 --enc-num-heads 1 --num-ref-points 8 --n 8000 --dataset physionet --seed 1 --save --norm --intensity --net hetvae_det --bound-variance --shuffle  --sample-tp 0.5 --elbo-weight 1.0 --mse-weight 0.0 --mixing interp_only --k-iwae 1
```
8. HeTVAE - HET - INT - DET - ALO (HTVAE mTAN)
```bash
python3 train.py --niters 2000 --lr 0.0001 --batch-size 128 --rec-hidden 32 --latent-dim 64 --width 512 --embed-time 128 --enc-num-heads 1 --num-ref-points 8 --n 8000 --dataset physionet --seed 1 --save --norm --intensity --net hetvae_det --bound-variance --shuffle  --sample-tp 0.5 --elbo-weight 1.0 --mse-weight 0.0 --mixing interp_only --k-iwae 1 --const-var --std 0.8
```