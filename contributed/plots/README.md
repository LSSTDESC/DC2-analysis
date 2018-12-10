# This is the directory that will contain the plots you make with DC2 data


Ideally the plots are produced from a notebook/code submitted in the directory, however you can also submit a plot on its own into this directory. In addition to the plot, you should submit text (e.g. a long figure caption) that describes the science case and pipeline and accompanies the figure.

In order to submit the figure/text, you can follow the same procedure as for submitting the analysis notebooks (repeated from the README one level up):


Here are the instructions to contribute your analysis notebooks/code to this repo.

If you are an experienced git user, the TL;DR is:
- Checkout a new branch.
- Put your plot into `contributed/plots`
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

5.  Add your work. If you already have a plot that is ready to share,
    you can simply copy it into `contributed/plots`.
    ```bash
    cd ./contributed/plots
    cp /path/to/your/awesome-plot ./
    cp /path/to/your/awesome-plot-description.txt ./
    git add awesome-plot
    git commit awesome-plot -m "add an awesome plot"
    git add awesome-plot-description.txt 	
    git commit awesome-plot-description.txt -m "add an awesome plot description"
    ```
7.  Commit and push to your forked repo
    ```bash
    git push origin u/username/short-description-about-your-work master
    ```

8.  Head back to https://github.com/LSSTDESC/DC2-analysis to create a pull request.

