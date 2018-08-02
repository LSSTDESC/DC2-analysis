# Contribute your analysis notebooks/code

Here are the instruction to contribute your analysis notebooks/code to this repo.

If you are an experienced git user, the TL;DR is:
- Checkout a new branch.
- Put your notebooks/code into `contributed/` (clear notebook output first).
- Commit, push, and submit a PR.

Otherwise, here's a step-by-step guide:

1.  If you don't have write permission for this repo, request it.
    To request write permission, [visit this "DC2-analysis-contributors" team page](https://github.com/orgs/LSSTDESC/teams/dc2-analysis-contributors/members)
    and click the "Request to join" button on the upper right corner.

2.  Clone this repo (most likely on NERSC because your notebook is probably
    sitting on NERSC or your development will involve accessing the DC2 data on NERSC).
    You only need to do this once.
    ```bash
    cd ~/desc # or another directory of your choice
    git clone git@github.com:LSSTDESC/DC2-analysis.git
    ```
    Note that if you haven't [added your SSH key to GitHub](https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/),
    you'll need to use HTTPS instead:
    ```bash
    git clone https://github.com/LSSTDESC/DC2-analysis.git
    ```

3.  Update the master branch (**always do this!**)
    ```bash
    cd ~/desc/DC2-analysis
    git checkout master
    git pull --ff-only
    ```

4.  Create a new branch for your work:
    ```bash
    git checkout -b u/username/short-description-about-your-work master
    ```
    Change `username` to your GitHub username and
    `short-description-about-your-work` to a very short description of your
    (planned) work. Use hyphens instead of spaces.

5.  Add your work. If you already have a notebook that is ready to share,
    you can simply copy it into
    `contributed/`.
    ```bash
    cd ./contributed
    cp /path/to/your/awesome-notebook.ipynb ./
    jupyter nbconvert --clear-output awesome-notebook.ipynb # see note below
    git add awesome-notebook.ipynb
    git commit -m "add an awesome notebook"
    ```
    Note that on NERSC, you'll probably need to replace `jupyter` with
    `/usr/common/software/python/3.6-anaconda-4.4/bin/jupyter` in the second command above.
    This clears the output of the notebook to make it git-friendly.

    On the other hand, if you'd like to start with a tutorial notebook and
    modify it, you should first *copy* the tutorial notebook to `contributed/`.
    Then you can use jupyter-dev to work on your notebook
    ([DESC members can find instruction here](https://confluence.slac.stanford.edu/x/Xgg4Dg)).
    When you can ready to commit, make sure you clear all output, hit the save button
    on the Jupyter interface, and then come back to command line to add and commit.

7.  Commit and push to your forked repo
    ```bash
    git push origin u/username/short-description-about-your-work master
    ```

8.  Head back to https://github.com/LSSTDESC/DC2-analysis to create a pull request.
