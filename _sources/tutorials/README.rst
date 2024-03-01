DC2 Tutorials
=============

This directory contains tutorial and demonstration notebooks convering how to access and use the DC2 datasets.
See the index table below for links to the notebook code, and an auto-rendered view of the notebook with outputs.

Notes for Tutorial Users
------------------------

*(Notes for tutorial contributors are at the bottom of the page.)*

* If you are unsure on exactly how to start to use these tutorials, you may want to look at
  this `step-by-step guide <https://confluence.slac.stanford.edu/x/Xgg4Dg>`_ *(requires Confluence login)*.

* If you want to obtain an overview of the different types of DC2 data products and their access methods,
  take a look at `DC2 Data Product Overview <https://confluence.slac.stanford.edu/x/oJgHDg>`_ *(requires Confluence login)*.

* Many of the tutorials below use the Generic Catalog Reader (GCR) to access the catalogs.
  To learn more about GCR, visit `LSSTDESC/gcr-catalogs <https://github.com/LSSTDESC/gcr-catalogs>`_ and see the information therein.

* You are very encouraged to use these tutorials as a starting point for your own projects!
  And you can contribute your analysis notebooks following `the instruction here <https://github.com/LSSTDESC/DC2-analysis/blob/master/contributed/README.md>`_.
  It is expected that some of your analysis notebooks would be based on these tutorials
  and hence have duplicated code snippets.
  You don't need to worry about that for now;
  we can always identify commonly used pieces and refactor them as standalone tools at a later time.


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


   * - Object catalog GCR Tutorial Part IV: Photo-z information
     - Use the GCR to access the Photo-z information that are provided as an "add-on" to the object catalog
     - `ipynb <object_gcr_4_photoz.ipynb>`_, `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/object_gcr_4_photoz.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/object_gcr_4_photoz.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/object_gcr_4_photoz.log

     - `Yao-Yuan Mao <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@yymao>`_, `Sam Schmidt <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@sschmidt23>`_


   * - Object catalog with Spark
     - Introduction of using Spark to access the object catalogs
     - `ipynb <object_spark_1_intro.ipynb>`_, `rendered <https://nbviewer.jupyter.org/github/LSSTDESC/DC2-analysis/blob/rendered/tutorials/object_spark_1_intro.nbconvert.ipynb>`_

       .. image:: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/object_spark_1_intro.svg
          :target: https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/log/object_spark_1_intro.log

     - `Julien Peloton <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@JulienPeloton>`_


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

   * - DC2 Run 2.1i DR1 Object Catalog
     - Preliminary Validation of DR1 Object Catalog
     - `ipynb <../validation/validate_dc2_run2.1i_object_table.ipynb>`_,
       `rendered <https://github.com/LSSTDESC/DC2-analysis/blob/rendered/validation/validate_dc2_run2.1i_object_table.ipynb>`_
     - `Michael Wood-Vasey <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@wmwv>`_

   * - Difference Image Analysis - Stamps and Lightcurves
     - Exploring a test Run 1.2p DIA run on (tract, patch) = (4849, '6,6')
     - `ipynb <dia_source_object_stamp.ipynb>`_,
       `rendered <https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/dia_source_object_stamp.ipynb>`_
     - `Michael Wood-Vasey <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@wmwv>`_

   * - Difference Image Analysis - Supernova Example
     - Comparing supernova lightcurves to variable+transient truth catalog for test Run 1.2p DIA run on (tract, patch) = (4849, '6,6')
     - `ipynb <dia_sn_vs_truth.ipynb>`_,
       `rendered <https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/dia_sn_vs_truth.ipynb>`_
     - `Michael Wood-Vasey <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@wmwv>`_

   * - Using PostgreSQL Introduction - Object table
     - Use object table as first example of how to access PostgreSQL database. Also includes mini SQL primer
     - `ipynb <postgres_object_1_intro.ipynb>`_,
       `rendered <https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/postgres_object_1_intro.ipynb>`_
     - `Joanne Bogart <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@JoanneBogart>`_

   * - PostgreSQL - Object table part 2
     - More advanced queries of object table
     - `ipynb <postgres_object_2.ipynb>`_,
       `rendered <https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/postgres_object_2.ipynb>`_
     - `Joanne Bogart <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@JoanneBogart>`_
     
   * - PostgreSQL - Forced source
     - Use PostgreSQL fourcedsource view to access forced source, plot light curves
     - `ipynb <postgres_forcedsource.ipynb>`_,
       `rendered <https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/postgres_forcedsource.ipynb>`_
     - `Joanne Bogart <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@JoanneBogart>`_
     
   * - PostgreSQL - Truth data
     - Use PostgreSQL to access star truth summary and variability tables
     - `ipynb <postgres_truth.ipynb>`_,
       `rendered <https://github.com/LSSTDESC/DC2-analysis/blob/rendered/tutorials/postgres_truth.ipynb>`_
     - `Joanne Bogart <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@JoanneBogart>`_
----

Notes for Tutorial Contributors
-------------------------------
Both tutorial and demo notebooks are hugely useful resources - pull requests are most welcome! A detailed instruction can be found `here <https://github.com/LSSTDESC/DC2-analysis/blob/master/contributed/README.md>`_.

* Before you commit a notebook, please make sure that a) it runs to completion and b) the outputs are cleared (to avoid both repo bloat and conflicts on every run due to semantically equivalent but bitwise-distinct output blobs).

* Please update the index table above, carefully copying and adapting the URLs, and making sure that all the lines in the table are correctly aligned (or else the table will not display). *Pro-tip: use the "preview changes" tab in the online GitHub editor to check that the table is correct before committing.*  Adding your notebook to the table will trigger the automatic testing of it once your PR is merged (see the "Semi-continuous Integration" section below).

* The "owner" of a notebook (that's you, as its contributor!) is responsible for accepting proposed modifications to it (by collaboration), and making sure that it does not go stale (by fixing issues posted about it).

* Every tutorial notebook needs an owner/last verified header, a statement of its goals (learning objectives) in the first markdown cell, and enough explanatory markdown (with links to docs, papers etc) to make the notebook make sense.

* Before August 2018, these tutorials were developed in the `DC2-production <https://github.com/LSSTDESC/DC2-production>`_ repo.You can `follow this link <https://github.com/LSSTDESC/DC2-production/search?q=label%3ATutorial&type=Issues>`_ to see issues and PRs that were related to these tutorials before they being moved here.


Semi-continuous Integration
---------------------------
All the notebooks listed in the table above (and on the master branch) can be run on Cori using the `beavis-ci <https://github.com/LSSTDESC/beavis-ci>`_ script, which then pushes them to an orphan "rendered" branch so that the outputs can be viewed.  Our ideal is that this will be run automatically daily, but that is not currently active.

    If the link to a rendered notebook yields a 404 error, please check the corresponding log file (by clicking on the "build:failing" badge) and issue the notebook's owner. If it looks like something has gone wrong overall (like, none of the notebook builds are passing, or the logs indicate some problem with the run environment, `issue @drphilmarshall <https://github.com/LSSTDESC/DC2-analysis/issues/new?body=@drphilmarshall>`_.
