# a quick hack to make the new stack backward compatabie 
import sys
try:
    import lsst.obs.lsst.phosim
except ImportError:
    print('[WARNING] You are using an older kernel. Log in to cori.nersc.gov and run\n\n/global/common/software/lsst/common/miniconda/kernels/setup.sh')
else:
    sys.modules['lsst.obs.lsstCam.lsstCamMapper'] = lsst.obs.lsst.phosim

__all__ = ['REPOS']    

# current available runs
REPOS = {
    '1.1p': '/global/cscratch1/sd/desc/DC2/data/Run1.1p/Run1.1/output',
    '1.2p': '/global/cscratch1/sd/desc/DC2/data/Run1.2p/w_2018_30/rerun/coadd-all2',
    '1.2i': '/global/cscratch1/sd/desc/DC2/data/Run1.2i/rerun/multiband',
}
