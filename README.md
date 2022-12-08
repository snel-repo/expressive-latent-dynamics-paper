# Expressive architectures enhance interpretability of dynamics-based neural population models
Experiments that probe the relationship between expressivity and interpretability in models of neural population dynamics.

If you find this code useful in your research, please cite the accompanying preprint:

>Sedler, A, Versteeg, C, Pandarinath, C. "Expressive architectures enhance interpretability of dynamics-based neural population models". arXiv 2022. https://arxiv.org/abs/2212.03771.

[![arXiv](https://img.shields.io/badge/arXiv-2212.03771-b31b1b.svg)](https://arxiv.org/abs/2212.03771)

## Codebase Structure
- `paper_src` - Source code that defines core dataset generation, model architecture, training, and evaluation functionality
- `configs` - [`hydra`](https://hydra.cc/)-composable config files that define data, model, and training hyperparameters
- `datasets` - Synthetic HDF5 datasets generated using `paper_src.data.ChaoticDataModule`
- `runs` - Checkpoints, training logs, and hyperparameters for models used in paper figures (Note: some files, including Tensorboard logs, have been removed to conserve space)
- `scripts/analysis` - Scripts that generate paper figures using trained models
- `scipts/training` - Scripts that retrain single models or multiple models in parallel using [`ray.tune`](https://docs.ray.io/en/latest/tune/index.html)

## Installation
After cloning the repo, install the codebase in a `conda` environment and use `pip` to install the `paper_src` package and its dependencies.
```
conda create -n expressive-paper python=3.9
cd expressive-latent-dynamics-paper
pip install -e .
```

## Reproducing Figures Using Trained Models
To reproduce figures from the paper, move to the `scripts/analysis` directory and run the following scripts:
- `1_plot_dataset.py` - Figure 1a
- `2_plot_metrics.py` - Figures 1c, 2a, and 2c
- `3_plot_trajs_fps.py` - Figures 1d, 2b, 2d, and 3a
- `4_plot_eigvals.py` - Figures 3b-e

## Training Models from Scratch
To train a single new model, move to the `scripts/analysis` directory and run the `train_single.py` script. This script will compose a configuration from the base config at `configs/single.yaml`. To overwrite the defaults defined in this config, you may edit the config itself or specify overrides directly to the train function. Model checkpoints and training logs will be stored at `runs/user_runs/single`.

To train many models in parallel, use the `train_multi.py` script. The script is set up to train all model-dataset combinations used in the paper (N=210), on multiple GPUs, with several models per GPU. Please refer to the [docs](https://docs.ray.io/en/latest/tune/index.html) for more information about the `ray.tune` API.

Note that `run_multi.py` script will put all of the runs in the same folder, which will need to be loaded slightly differently to be compatible with the analysis pipeline. However, loading from a single folder should be similar to, but easier than, loading from several subfolders.

## Open-source Acknowledgments
- Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. arXiv; 2019.
- Falcon, W., & The PyTorch Lightning team. (2019). PyTorch Lightning (Version 1.4) [Computer software]. https://doi.org/10.5281/zenodo.3828935
- Liaw R, Liang E, Nishihara R, Moritz P, Gonzalez JE, Stoica I. Tune: A research platform for distributed model selection and training. arXiv preprint arXiv:180705118 2018;.
- Gilpin W. Chaos as an interpretable benchmark for forecasting and data-driven modelling. Advances in Neural Information Processing Systems 2021;http://arxiv.org/abs/2110.05266.
- Yadan H. Hydra - A framework for elegantly configuring complex applications. GitHub; 2019. https://github.com/facebookresearch/hydra
- Poli M, Massaroli S, Yamashita A, Asama H, Park J. TorchDyn: A neural differential equations library. arXiv preprint arXiv:200909346 2020;.
- Golub MD, Sussillo D. FixedPointFinder: A Tensorflow toolbox for identifying and characterizing fixed points in recurrent neural networks. Journal of Open Source Software 2018;3(31):1003. https://doi.org/10.21105/joss.01003.
