#! /usr/bin/env bash
#=======================================================================
#+
# NAME:
#   beavis-ci.sh
#
# PURPOSE:
#   Enable occasional integration and testing. Like travis-ci but dumber.
#
# COMMENTS:
#   Makes "rendered" versions of all the notebooks listed in the README.rst 
#   and deploys them to a "rendered" orphan branch, pushed to GitHub for web display.
#
# INPUTS:
#
# OPTIONAL INPUTS:
#   -h --help     Print this header
#   -a --all      Run all the notebooks in the folder instead of the ones in the README
#   -u --username GITHUB_USERNAME, defaults to the environment variable
#   -k --key      GITHUB_API_KEY, defaults to the environment variable
#   -b --branch   Test the notebooks in a dev branch. Outputs still go to "rendered"
#   -n --no-push  Only run the notebooks, don't deploy the outputs
#   --html        Make html outputs instead
#
# OUTPUTS:
#
# EXAMPLES:
#
#   ./beavis-ci.sh -u $GITHUB_USERNAME -k $GITHUB_API_KEY
#
#-
# ======================================================================

HELP=0
no_push=0
html=0
all=0
src="$0"
branch='master'

while [ $# -gt 0 ]; do
    key="$1"
    case $key in
        -h|--help)
            HELP=1
            ;;
        -n|--no-push)
            no_push=1
            ;;
        -a|--all)
            all=1
            ;;
        -u|--username)
            shift
            GITHUB_USERNAME="$1"
            ;;
        -k|--key)
            shift
            GITHUB_API_KEY="$1"
            ;;
        --html)
            html=1
            ;;
        -b|--branch)
            shift
            branch="$1"
            ;;
    esac
    shift
done

if [ $HELP -gt 0 ]; then
  more $src
  exit 1
fi

date
echo "Welcome to beavis-ci: occasional integration and testing"

if [ $no_push -eq 0 ]; then
  if [ -z $GITHUB_USERNAME ] || [ -z $GITHUB_API_KEY ]; then
      echo "No GITHUB_API_KEY and/or GITHUB_USERNAME set, giving up."
      exit 1
  else
    echo "with deployment via GitHub token $GITHUB_API_KEY and username $GITHUB_USERNAME"
  fi
fi

echo "Cloning the DC2_Repo into the .beavis workspace:"

# Check out a fresh clone in a temporary hidden folder, over-writing
# any previous edition:
\rm -rf .beavis ; mkdir .beavis ; cd .beavis
git clone git@github.com:LSSTDESC/DC2_Repo.git
cd DC2_Repo/Notebooks
git checkout $branch

if [ $html -gt 0 ]; then
    echo "Making static HTML pages from the master branch notebooks:"
    outputformat="HTML"
    ext="html"
    target="html"
else
    echo "Rendering the master branch notebooks:"
    outputformat="notebook"
    ext="nbconvert.ipynb"
    target="rendered"
fi
mkdir -p log
webdir="https://github.com/LSSTDESC/DC2_Repo/tree/${branch}/Notebooks"

# Some notebooks have whitespace in their names - fix this first:
for notebook in *.ipynb; do
    ipynbfile="$( echo "$notebook" | sed s/' '/'_'/g )"
    if [ $ipynbfile != "$notebook" ]; then
        mv -f "$notebook" $ipynbfile
    fi
done

# Which notebooks to run? Either all of them, or the ones listed in README.rst:
if [ $all -eq 0 ]; then
    notebooks="$( grep '`ipynb' README.rst |& grep -v "No such file" | cut -d"<" -f2 | cut -d">" -f1 | sed s/'%20'/'_'/g | sed s/'%3A'/':'/g )" 
fi
# If README.rst is not present or empty of ipynb instances, default to running everything:
if [ $all -gt 0 ] || [ ${#notebooks[0]} -eq 0 ]; then
    notebooks="$( ls *.ipynb )"
fi

echo "$notebooks"

# Now loop over notebooks, running them one by one:
declare -a outputs
for notebook in $notebooks; do

    # Rename files to make them easier to work with:
    stem=$PWD/log/$notebook
    logfile=${stem%.*}.log
    svgfile=${stem%.*}.svg

    # Check the notebook exists:
    if [ -e "$notebook" ]; then
        echo "Running nbconvert on $notebook ..."    
    else
        echo "WARNING: could not find $notebook, setting build status to unknown."
        echo "$notebook: No such file in $branch branch" > $logfile
        cp ../../../.badges/unknown.svg $svgfile
        continue
    fi

    # Run the notebook:
    /usr/common/software/python/3.6-anaconda-4.4/bin/jupyter nbconvert \
                      --ExecutePreprocessor.kernel_name=desc-stack \
                      --ExecutePreprocessor.timeout=600 --to $outputformat \
                      --execute $notebook &> $logfile
    
    # Set the build status according to the output:
    output=${notebook%.*}.$ext

    if [ -e $output ]; then
        outputs+=( $output )
        echo "SUCCESS: $output produced."
        cp ../../../.badges/passing.svg $svgfile
    else
        echo "WARNING: $output was not created, read the log in $logfile for details."
        cp ../../../.badges/failing.svg $svgfile
    fi
    
done

if [ $no_push -gt 0 ]; then
    sleep 0

else
    echo "Attempting to push the rendered outputs to GitHub in an orphan branch..."

    cd ../
    git branch -D $target >& /dev/null
    git checkout --orphan $target
    git rm -rf .
    cd Notebooks
    git add -f "${outputs[@]}"
    git add -f log
    git commit -m "pushed rendered notebooks and log files"
    git push -q -f \
        https://${GITHUB_USERNAME}:${GITHUB_API_KEY}@github.com/LSSTDESC/DC2_Repo  $target
    echo "Done!"
    git checkout master

    echo ""
    echo "Please read the above output very carefully to see that things are OK. To check we've come back to our starting point correctly, here's a git status:"
    echo ""

    git status

fi

echo "beavis-ci finished: view the results at "
echo "   $webdir   "

cd ../../
date

# ======================================================================
