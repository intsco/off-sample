# Off-Sample Ion Image Detection in Imaging Mass Spectrometry

Project [Metaspace](https://metaspace2020.eu) ion image classification and segmentation.

# Requirements

* Ubuntu >= 14.04
* Conda >= 4.5.11
* Ideally Nvidia GPU

# Classification

## Setup

* Clone repository
* Create conda environment

```
cd classification
conda env create -n off-sample --file off-sample-env.yml
conda activate off-sample
```

* If Jupyter is already installed add a new kernel

```
python -m ipykernel install --user --name off-sample --display-name "off-sample"
```

* Otherwise install Jupyter into the environment.

## Run

* Start Jupyter and open `cnn_only_cv_on_gs.ipynb`
