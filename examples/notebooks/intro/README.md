# Data Prep Kit Introduction

This is an example featuring some of the features of data prep kit.

## Running the code

The code can be run on either 

1.  Google colab: very easy to run; no local setup needed.
2.  On your local Python environment.  Here is a quick guide.  You can  find instructions for latest version [here](../../../README.md#-getting-started)

```bash
conda create -n dpk-022dev1-py311 -y python=3.11
conda activate dpk-022dev1-py311

# install the following in 'data-prep-kit' environment
pip3 install --extra-index-url https://testpypi.python.org/pypi 'data-prep-toolkit-transforms==0.2.2.dev1' 
# pip3 install --extra-index-url https://testpypi.python.org/pypi 'data-prep-toolkit-transforms==0.2.2.dev1' 'data-prep-toolkit-transforms[ray]==0.2.2.dev1'
pip3 install jupyterlab   ipykernel  ipywidgets  humanfriendly

## install custom kernel
## Important: Use this kernel when running example notebooks!
python -m ipykernel install --user --name=data-prep-kit --display-name "dataprepkit"

# start jupyter and run the notebooks with this jupyter
jupyter lab
```

## Intro

This notebook will demonstrate processing PDFs

`PDFs ---> text ---> chunks --->   exact dedupe ---> fuzzy dedupe ---> embeddings`

[python version](dpk_intro_1_python.ipynb)  &nbsp;   |   &nbsp;  [ray version](dpk_intro_1_ray.ipynb)
