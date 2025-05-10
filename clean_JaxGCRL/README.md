# Clean-JaxGCRL

JaxGCRL code ported to [cleanrl](https://github.com/vwxyzjn/cleanrl) style implementations. To optimize for both speed and ease of understanding. Comes at cost of duplicate / redundant code.

## Installation

```bash
CONDA_OVERRIDE_CUDA="12.0" conda create --name expl-env python=3.10 numpy==1.26.4 jax==0.4.23 "jaxlib==0.4.23=cuda120*" flax==0.7.4 -c conda-forge -c nvidia -y
pip install tyro wandb==0.17.9 wandb_osh==1.2.2  brax==0.10.1 mediapy==1.2.2 scipy==1.12.0
```

```bash
CONDA_OVERRIDE_CUDA="12.0" conda create --name expl-env-jupyter python=3.10 numpy==1.26.4 jax==0.4.23 "jaxlib==0.4.23=cuda120*" flax==0.7.4 ipykernel jupyter -c conda-forge -c nvidia -y
python -m ipykernel install --user --name=expl-env-jupyter
pip install tyro wandb==0.17.9 wandb_osh==1.2.2  brax==0.10.1 mediapy==1.2.2 scipy==1.12.0
```


## jax version 0.4.31
```bash
CONDA_OVERRIDE_CUDA="12.0" conda create --name expl-env-jax0431-jupyter python=3.10 numpy==1.26.4 jax==0.4.31 "jaxlib==0.4.31=cuda120*" flax==0.7.4 ipykernel jupyter -c conda-forge -c nvidia -y
python -m ipykernel install --user --name=expl-env-jax0431-jupyter
pip install tyro wandb==0.17.9 wandb_osh==1.2.2  brax==0.10.1 mediapy==1.2.2 scipy==1.12.0
```


```bash
conda create --name expl-env-arm python=3.10
pip install numpy==1.26.4 jax[cuda12_pip]==0.4.23 jaxlib==0.4.23 flax==0.7.4 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tyro wandb==0.17.9 wandb_osh==1.2.2  brax==0.10.1 mediapy==1.2.2 scipy==1.12.0
```



## expl-env (but additional pip installs for JaxGCRL baselines)
```bash
CONDA_OVERRIDE_CUDA="12.0" conda create --name expl-env-JaxGCRL python=3.10 numpy==1.26.4 jax==0.4.23 "jaxlib==0.4.23=cuda120*" flax==0.7.4 -c conda-forge -c nvidia -y
pip install tyro wandb==0.17.9 wandb_osh==1.2.2  brax==0.10.1 mediapy==1.2.2 scipy==1.12.0
```
can 


## expl-env but flax==0.7.5 so that can do flax.linen.scan
```bash
CONDA_OVERRIDE_CUDA="12.0" conda create --name expl-env-flax075 python=3.10 numpy==1.26.4 jax==0.4.23 "jaxlib==0.4.23=cuda120*" flax==0.7.5 -c conda-forge -c nvidia -y
pip install tyro wandb==0.17.9 wandb_osh==1.2.2  brax==0.10.1 mediapy==1.2.2 scipy==1.12.0
```

## expl-env but flax==0.7.5 so that can do flax.linen.scan + jupyter
```bash
CONDA_OVERRIDE_CUDA="12.0" conda create --name expl-env-flax075-jupyter python=3.10 numpy==1.26.4 jax==0.4.23 "jaxlib==0.4.23=cuda120*" flax==0.7.5 ipykernel -c conda-forge -c nvidia -y
python -m ipykernel install --user --name=expl-env-flax075-jupyter
pip install tyro wandb==0.17.9 wandb_osh==1.2.2  brax==0.10.1 mediapy==1.2.2 scipy==1.12.0
```


## expl-env but flax==0.9.0 so that can do flax.linen.scan + jupyter
```bash
CONDA_OVERRIDE_CUDA="12.0" conda create --name expl-env-flax090-jupyter python=3.10 numpy==1.26.4 jax==0.4.23 "jaxlib==0.4.23=cuda120*" flax==0.9.0 ipykernel -c conda-forge -c nvidia -y
python -m ipykernel install --user --name=expl-env-flax090-jupyter
pip install tyro wandb==0.17.9 wandb_osh==1.2.2  brax==0.10.1 mediapy==1.2.2 scipy==1.12.0
```


## expl-env but flax==0.8.5 so that can do flax.linen.scan + jupyter
```bash
CONDA_OVERRIDE_CUDA="12.0" conda create --name expl-env-flax085-jupyter python=3.10 numpy==1.26.4 jax==0.4.23 "jaxlib==0.4.23=cuda120*" flax==0.8.5 ipykernel -c conda-forge -c nvidia -y
python -m ipykernel install --user --name=expl-env-flax085-jupyter
pip install tyro wandb==0.17.9 wandb_osh==1.2.2  brax==0.10.1 mediapy==1.2.2 scipy==1.12.0
```