DC2 Tutorials
=============

This directory contains tutorial and demonstration notebooks convering how to access and use the DC2 datasets.
See the index table below for links to the notebook code, and an auto-rendered view of the notebook with outputs.

If you are unsure on exactly how to start to use these tutorials, you may want to look at 
`this step-by-step guide on Confluence <https://confluence.slac.stanford.edu/x/Xgg4Dg>`_ *(DESC members only)*.

Notes on how to contribute more notebooks, and how the rendering is made, are at the bottom of the page.

.. list-table::
   :widths: 10 20 10 10
   :header-rows: 1

   * - Notebook
     - Short description
     - Links
     - Owner


   * - Object catalog GCR Tutorial Part I: GCR access
     - Use the GCR for simple access to the object catalogs
     - `ipynb <object_gcr_1_intro.ipynb>`_, `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/object_gcr_1_intro.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/object_gcr_1_intro.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/object_gcr_1_intro.log

     - `Francois Lanusse <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@EiffL>`_, `Javier Sanchez <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@fjaviersanchez>`_


   * - Object catalog GCR Tutorial Part II: Lensing Cuts
     - Use the GCR to access the object catalog and build a lensing sample similar to the HSC Y1 shape catalog
     - `ipynb <object_gcr_2_lensing_cuts.ipynb>`_, `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/object_gcr_2_lensing_cuts.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/object_gcr_2_lensing_cuts.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/object_gcr_2_lensing_cuts.log

     - `Francois Lanusse <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@EiffL>`_, `Javier Sanchez <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@fjaviersanchez>`_


   * - Object catalog GCR Tutorial Part III: Guided Challenges
     - Use the GCR to access the object catalog and solve some typical data analysis problems
     - `ipynb <object_gcr_3_challenges.ipynb>`_, `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/object_gcr_3_challenges.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/object_gcr_3_challenges.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/object_gcr_3_challenges.log

     - `Francois Lanusse <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@EiffL>`_, `Javier Sanchez <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@fjaviersanchez>`_


   * - Object catalog with pandas, color-color stellar locus
     - Directly access the Run 1.1p object catalogs using pandas and explore the stellar locus
     - `ipynb <object_pandas_stellar_locus.ipynb>`_,
       `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/object_pandas_stellar_locus.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/object_pandas_stellar_locus.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/object_pandas_stellar_locus.log

     - `Michael Wood-Vasey <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@wmwv>`_


   * - DM DeepCoadd Catalog with Butler Access: lensing cuts
     - Use the Butler to build a lensing sample similar to the HSC Y1 shape catalog
     - `ipynb <dm_butler_lensing_cuts.ipynb>`_,
       `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/dm_butler_lensing_cuts.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/dm_butler_lensing_cuts.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/dm_butler_lensing_cuts.log

     - `Jim Chiang <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@jchiang87>`_


   * - DM Postage Stamps with Butler
     - Make some small cutout images and visualize them
     - `ipynb <dm_butler_postage_stamps.ipynb>`_,
       `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/dm_butler_postage_stamps.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/dm_butler_postage_stamps.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/dm_butler_postage_stamps.log

     - `Michael Wood-Vasey <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@wmwv>`_


   * - Plotting skymap in DM catalogs using Butler
     - Use the data butler to obtain information on the skyMap used in the coadd analyses performed by the DRP pipeline.
     - `ipynb <dm_butler_skymap.ipynb>`_,
       `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/dm_butler_skymap.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/dm_butler_skymap.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/dm_butler_skymap.log

     - `Jim Chiang <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@jchiang87>`_


   * - Matching catalogs using the LSST Stack matching code
     - Spatial matching of objects using the DM Stack
     - `ipynb <matching_stack.ipynb>`_,
       `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/matching_stack.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/matching_stack.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/matching_stack.log

     - `Jim Chiang <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@jchiang87>`_


   * - Matching catalogs using FoF algorithm
     - Using the Friends-of-Friends algorithm to match the extragalactic, truth, and object catalogs
     - `ipynb <matching_fof.ipynb>`_,
       `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/matching_fof.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/matching_fof.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/matching_fof.log

     - `Yao-Yuan Mao <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@yymao>`_


   * - Truth Catalog with GCR
     - Example of accessing DC2 truth catalog with GCR
     - `ipynb <truth_gcr_intro.ipynb>`_,
       `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/truth_gcr_intro.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/truth_gcr_intro.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/truth_gcr_intro.log

     - `Scott Daniel <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@danielsf>`_


   * - Truth Catalog with GCR: variables and transients
     - Example of accessing variables and transient objects in the truth catalog with GCR
     - `ipynb <truth_gcr_variables.ipynb>`_,
       `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/truth_gcr_variables.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/truth_gcr_variables.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/truth_gcr_variables.log

     - `Yao-Yuan Mao <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@yymao>`_, 
       `Scott Daniel <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@danielsf>`_


   * - Extragalactic catalog with GCR: redshift distributions
     - Extract, plot and explore the differential number counts of galaxies
     - `ipynb <extragalactic_gcr_redshift_dist.ipynb>`_,
       `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/extragalactic_gcr_redshift_dist.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/extragalactic_gcr_redshift_dist.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/extragalactic_gcr_redshift_dist.log

     - `Eve Kovacs <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@evevkovacs>`_


   * - Extragalactic catalog with GCR: Halo Occupation Distribution
     - Compute and plot the HOD for the extragalactic catalog
     - `ipynb <extragalactic_gcr_hod.ipynb>`_,
       `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/extragalactic_gcr_hod.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/extragalactic_gcr_hod.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/extragalactic_gcr_hod.log

     - `Yao-Yuan Mao <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@yymao>`_


   * - Extragalactic catalog with GCR: mass relations
     - Compute and plot the relations between halo mass and other quantities in the extragalactic catalog
     - `ipynb <extragalactic_gcr_mass_relations.ipynb>`_,
       `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/extragalactic_gcr_mass_relations.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/extragalactic_gcr_mass_relations.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/extragalactic_gcr_mass_relations.log

     - `Yao-Yuan Mao <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@yymao>`_


   * - Extragalactic catalog with GCR: cluster colors
     - Access the extragalactic catalog with the GCR, and explore colors in galaxy clusters
     - `ipynb <extragalactic_gcr_cluster_colors.ipynb>`_,
       `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/extragalactic_gcr_cluster_colors.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/extragalactic_gcr_cluster_colors.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/extragalactic_gcr_cluster_colors.log

     - `Dan Korytov <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@dkorytov>`_


   * - Extragalactic catalog with GCR: cluster members
     - Extract cluster member galaxies from the extragalactic catalog and plot them on the sky
     - `ipynb <extragalactic_gcr_cluster_members.ipynb>`_,
       `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/extragalactic_gcr_cluster_members.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/extragalactic_gcr_cluster_members.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/extragalactic_gcr_cluster_members.log

     - `Dan Korytov <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@dkorytov>`_


----

Notes for Contributors
----------------------
Both tutorial and demo notebooks are hugely useful resources - pull requests are most welcome!

* Before you commit a notebook, please make sure that a) it runs to completion and b) the outputs are cleared (to avoid both repo bloat and conflicts on every run due to semantically equivalent but bitwise-distinct output blobs).

* Please update the index table above, carefully copying and adapting the URLs, and making sure that all the lines in the table are correctly aligned (or else the table will not display). *Pro-tip: use the "preview changes" tab in the online GitHub editor to check that the table is correct before committing.*  Adding your notebook to the table will trigger the automatic testing of it once your PR is merged (see the "Semi-continuous Integration" section below).

* The "owner" of a notebook (that's you, as its contributor!) is responsible for accepting proposed modifications to it (by collaboration), and making sure that it does not go stale (by fixing issues posted about it).

* Every tutorial notebook needs an owner/last verified header, a statement of its goals (learning objectives) in the first markdown cell, and enough explanatory markdown (with links to docs, papers etc) to make the notebook make sense.

* Before August 2018, these tutorials were developed in the [DC2-production](https://github.com/LSSTDESC/DC2-production) repo.You can [follow this link](https://github.com/LSSTDESC/DC2-production/search?q=label%3ATutorial&type=Issues) to see issues and PRs that were related to these tutorials before they being moved here. 


Semi-continuous Integration
---------------------------
All the notebooks listed in the table above (and on the master branch) are run every 6 hours on Cori using the [`beavis-ci` script](beavis-ci.sh), which then pushes them to an orphan "rendered" branch so that the outputs can be viewed. (At present, it seems that `DC2-analysis` admin permissions are needed to execute this push, but in principle anyone could run this script.)

    If the link to a rendered notebook yields a 404 error, please check the corresponding log file (by clicking on the "build:failing" badge) and issue the notebook's owner. If it looks like something has gone wron with the cron job (like, none of the notebook builds are passing, or the logs indicate some problem with the run environment, `issue @drphilmarshall <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@drphilmarshall>`_.
