import numpy as np
from scipy.stats import skew, kurtosis
import healpy as hp

def binned_statistic(x, values, func, nbins, range):
    '''The usage is approximately the same as the scipy one
    from https://stackoverflow.com/questions/26783719/effic
    iently-get-indices-of-histogram-bins-in-python'''
    from scipy.sparse import csr_matrix
    r0, r1 = range
    mask = (x > r0) &  (x < r1)
    x = x[mask]
    values = values[mask]
    N = len(values)
    digitized = (float(nbins) / (r1-r0) * (x-r0)).astype(int)
    S = csr_matrix((values, [digitized, np.arange(N)]), shape=(nbins, N))
    return np.array([func(group) for group in np.split(S.data, S.indptr[1:-1])])

def create_map(ra,dec,mask,nside=4096):
    pix_nums = hp.ang2pix(nside,np.pi/2-dec*np.pi/180.,np.pi/180*ra)
    goodpix = np.in1d(pix_nums,np.where(mask>0)[0])
    map_gal = np.bincount(pix_nums[goodpix],minlength=12*nside**2)
    return map_gal

def change_resol(mapin,mask,nsideout):
    arr_out = np.zeros(12*nsideout**2)
    th, phi = hp.pix2ang(hp.get_nside(mapin),np.where(mask>0)[0])
    pix_nums = hp.ang2pix(nsideout,th,phi)
    bin_count = np.bincount(pix_nums,weights=mapin[np.where(mask>0)[0]],minlength=12*nsideout**2)
    return bin_count

def cic_analysis(data,mask,mask_comp=0.9,nboot=20,nameout=None):
    ra = data['ra']
    dec = data['dec']
    mask = hp.ud_grade(mask,nside_out=2048)
    m0 = create_map(ra,dec,mask,nside=2048)
    map_arr = [change_resol(m0,mask,32*2**i) for i in range(0,6)]
    sigma = np.zeros(len(map_arr))
    skw = np.zeros(len(map_arr))
    kurt = np.zeros(len(map_arr))
    dsigma = np.zeros(len(map_arr))
    dskw = np.zeros(len(map_arr))
    dkurt = np.zeros(len(map_arr))
    Nup = np.zeros(len(map_arr))
    scale_arr = np.zeros(len(map_arr))
    for isize in range(0,len(map_arr)):
        map_index = isize
        nside = hp.get_nside(map_arr[map_index])
        new_mask = hp.ud_grade(mask,nside_out=nside)
        scale_arr[isize] = np.sqrt(hp.nside2pixarea(nside, degrees=True))
        masked = new_mask > mask_comp
        Ngal = np.sum(1.0*map_arr[map_index][masked]/new_mask[masked])
        Npix = np.count_nonzero(masked)
        Nup[isize] = Npix
        avdens = Ngal/(1.0*Npix)/hp.nside2pixarea(nside) 
        delta_map = (1.0*map_arr[map_index][masked]/new_mask[masked])/hp.nside2pixarea(nside)/avdens-1
        avdens = avdens*hp.nside2pixarea(nside)
        sigma[isize] = np.mean(delta_map**2)-1./avdens
        skw[isize] = np.mean(delta_map**3)-3./avdens*sigma[isize]-1/avdens**2
        kurt[isize] = np.mean(delta_map**4)-3*np.mean(delta_map**2)**2-7/avdens**2*sigma[isize]-6/avdens*skw[isize]-1/avdens**3
        if(nboot>len(np.where(masked==True)[0])):
            nbb=len(np.where(masked==True)[0])
        else:
            nbb=nboot
        if nbb>0:
            sigma_b = np.std(np.random.choice(delta_map,(len(delta_map),nbb)),axis=1) 
            skw_b = skew(np.random.choice(delta_map,(len(delta_map),nbb)),axis=1)
            kurt_b = kurtosis(np.random.choice(delta_map,(len(delta_map),nbb)),axis=1)
            dsigma[isize]=np.std(sigma_b**2)
            dskw[isize]=np.std(skw_b)
            dkurt[isize]=np.std(kurt_b)
        else:
            dsigma[isize]=0.
            dskw[isize]=0.
            dkurt[isize]=0.
    kurt = kurt/sigma**3
    skw = skw/sigma**2
    if nameout is not None:
        tab_out_1 = astropy.table.Table([sigma,dsigma,skw,dskw,kurt,dkurt],
                                    names=('sigma','dsigma','skw','dskw','kurt','dkurt'))
        tab_out_1.write(nameout+'.fits.gz',overwrite=True)
    return sigma,dsigma,skw,dskw,kurt,dkurt,scale_arr
