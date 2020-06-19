# Introduction

This directory contains notebooks related to the validation work
being performed for the different DC2 runs. 

These tutorials are intended to be run at NERSC using the [JupyterHub environment](https://jupyter.nersc.gov). If you are unsure
on how to proceed, please go [here](https://github.com/LSSTDESC/DC2-analysis/tree/master/tutorials).

You are encouraged to use the code in these notebooks for your own analysis.

### Notebooks


| Notebook | Short Description | Links | Owner |
|----------|-------------------|-------|-------|
|DC2 QA | Validation tests on source catalogs and calexps | [ipynb](./DC2_calexp_src_validation_1p2.ipynb) | [Javier Sanchez](https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@fjaviersanchez) |
|PSF ellipticity tests | Measuring PSF ellipticity in DC2 calexps | [ipynb](./Run_1.2p_PSF_tests.ipynb) | [Javier Sanchez](https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@fjaviersanchez) |
|DC2 Run 2.1i Object | Validation on Run 2.1i Object Catalog | [ipynb](validate_dc2_run2.1i_object_table.ipynb), [rendered](https://github.com/LSSTDESC/DC2-analysis/blob/rendered/validation/validate_dc2_run2.1i_object_table.ipynb) | [Michael Wood-Vasey](https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@wmwv) |


The code in these notebooks can be, in principle, ported to other DC2 runs.

Please, if you find any problems feel free to open a new issue in this repository and let the notebook's owner(s) know.


### Scripts

To run the validate_dc2_run2.2i_object_table.py script
first start an interactive session on a Node:
Running through an interaction session is the reasonable way
to run this on NERSC because we need 0.5-1 hour to run this,
but the 'debug' queue is only 0.5 hours,
and waiting for a regular queue slot can easily take at least a day.
We need an entire node for the memory (~128 GB)
```
salloc -N 1 -C haswell -q interactive --time=02:00:00
```

Once you're running on the allocated node:
```
module load python3
```
Or
```
start-kernel-cli.py desc-python
```

And then run the script, ideally pipeline the output to a log file for future reference:
```
python validate_dc2_run2.2i_object_table.py > validate_dc2_run2.2i_object_table.log 2>&1 < /dev/null &
```

#### Dask-based script
The instructions for running the Dask-based script are the same.  Once we're on our own node, we'll use the availalable processors for Dask.
```
python validate_dc2_run2.2i_object_table_dask.py > validate_dc2_run2.2i_object_table_dask.log 2>&1 < /dev/null &
```
