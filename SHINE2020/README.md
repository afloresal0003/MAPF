A repository for the ACT Lab's SHINE 2020 Research


## Installation Instructions

This section has instructions for installing the software and librariries needed for running the code in this repo. `conda` instructions for creating a new environment will be used and the packages will be intalled there

First, make sure Anaconda distribution is installed. Packages can be downloaded from the bottom of [this page](https://www.anaconda.com/products/individual).

Once Anaconda is installed on your computer, create a new environment like so:

`$ conda create -n usc python=3.7.5`

This will create a new environment called `usc` which can be activated with

`$ conda activate usc`

This repository needs `numpy`, `matplotlib` and PyTorch to run.
TO install the first two, use:

`$ conda install numpy matplotlib`

Pytorch needs this specific command when using MacOS:

`$ conda install pytorch torchvision -c pytorch`

For other operating systems check [this page](https://pytorch.org/get-started/locally/).

This is all we need to set it up. 
