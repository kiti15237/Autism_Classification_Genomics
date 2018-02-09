# CGT Regression

This directory contains the scripts and data for applying different regression algorithms to different tasks of predicting autism phenotype from genotype.  There are a few different kinds of files in this directory:

- *data* is a directory containing many of the reference data files necessary for running analyses
- *.py files contain helper code used for running analyses
- *.ipynb files contain highly annotated pipelines for running analyses

Two different kinds of analyses are featured here.  They are:

1. ASD diagnosis prediction from genotype
2. phenotype cluster prediction from genotype

NOTE: cluster prediction was a secondary goal of this project and the code for this aim has not been updated in some time.  In addition, notebooks pertaining to this task will run on sherlock, but will not run on any fork/clone of this github repository outside of sherlock because the required phenotype cluster label files are not in the data folder of this repository.
