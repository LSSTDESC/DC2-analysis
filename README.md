# DC2-analysis

The repository contains general analysis scripts for the DC2 simulated dataset.  Project-specific work can be done in separate repositories, but some analysis tasks will be generally useful, and the relevant scripts and notebooks can be included here for more general consumption.

:arrow_right: [Click here to browse rendered notebooks](https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/tree/rendered/)!

## Links

The [DC2-production repository](https://github.com/LSSTDESC/DC2-production) includes documents, discussion, and scripts related to the production and validation of DC2.  Please go there for more information about DC2.

If you find some problem or have a question about tutorials and scripts in this repository, please ask by [opening an issue](https://github.com/LSSTDESC/DC2-analysis/issues).

## License, credit

This repository has a BSD 3-clause license (see LICENSE).

The tutorials in the [tutorials directory](https://github.com/LSSTDESC/DC2-analysis/tree/master/tutorials) all indicate at what date they were last verified to run for the specific dataset that is used throughout the tutorial.  Additional contributions to the repository will include similar information, so that the state of a given piece of code can be clearly identified by users.

If you make use of the notebooks in this repository for your research, please give a link to this repository and an acknowledgement of the author(s) of the notebook.

## How to get involved

In the [tutorials directory](https://github.com/LSSTDESC/DC2-analysis/tree/master/tutorials) you can find a set of well-documented tutorials that show how to access and use DC2 data in a variety of ways.  The [README](https://github.com/LSSTDESC/DC2-analysis/tree/master/tutorials/README.rst) in that directory has a table describing the tutorials.

New scripts and notebooks can be added in the [contributed directory](https://github.com/LSSTDESC/DC2-analysis/tree/master/contributed).  The [README](https://github.com/LSSTDESC/DC2-analysis/tree/master/contributed/README.md) has a full description of how to do so.

## Notebook review

There are two reasons why one might add a notebook to the repository:

1) Adding a generally applicable notebook that you think should be added to the master set of notebooks for DC2 analysis (such as a tutorial or a validation notebook; we imagine the DC2 team will be adding most of these). These will be reviewed by another DC2 team member.
2) You have a notebook for your particular science case that you'd like to add (usually under `contributed`; the notebook could combine some tasks from the existing notebooks, or do something totally different etc). For these notebooks, we will engage the analysis teams for input.

Some suggestions for adding new notebooks (e.g. contribution type 2 above). The code should:

a) run without issues; 
b) have some explanatory text about what the notebook is trying to do; 
c) use recommended packages/instructions whenever possible (e.g., not skipping API to access underlying files) and if not, clearly document their atypical/unrecommended usage; and
d) follow reasonable coding style/practices (for some examples, see the existing tutorial notebooks; or see the [DESC Coding Guidelines](https://docs.google.com/document/d/1v54bVQI2NejK2UqACDnGXj1t6IGFgY3Uc1R7iV2uLpY/edit?usp=sharing)).


## Questions?

If the links above don't do the job, please [open an issue](https://github.com/LSSTDESC/DC2-analysis/issues) with your question.  Alternatively, the #desc-dc2-users channel on the LSSTC slack is a good place to talk with others who are analyzing DC2 data.
