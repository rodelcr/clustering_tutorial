import numpy as np

from os.path import dirname, abspath, join as pjoin

import Corrfunc

from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks

from Corrfunc.io import read_catalog

from Corrfunc.utils import convert_rp_pi_counts_to_wp


import matplotlib.pyplot as plt 

from astropy.table import Table

from scipy.ndimage import gaussian_filter1d

import matplotlib.colors as colors
from scipy import stats
from scipy.special import gamma

import pyccl as ccl
from scipy.integrate import quad, simpson
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.integrate import simpson as simps
from scipy import interpolate


little_h = 0.67
cosmo = ccl.Cosmology(Omega_c = 0.1198/little_h**2, Omega_b = 0.02233/little_h**2, h = little_h, sigma8 =  0.8101, n_s = 0.9652, matter_power_spectrum='linear')

# z = 0.75
# a = (1/(1.+z))
#xir = ccl.correlation_3d(cosmo, a, bin_center)





plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams.update({'figure.autolayout': True})


plt.rc('text', usetex=True)
plt.rc('font', family='serif',size=15)
plt.rc('axes', linewidth=1.5) # change back to 1.5
plt.rc('axes', labelsize=20) # change back to 10
plt.rc('xtick', labelsize=22, direction='in')
plt.rc('ytick', labelsize=22, direction='in')
plt.rc('legend', fontsize=15) # change back to 7

# setting xtick parameters:

plt.rc('xtick.major',size=10,pad=4)
plt.rc('xtick.minor',size=5,pad=4)

plt.rc('ytick.major',size=10)
plt.rc('ytick.minor',size=5)


def norberg_error_local(outs_):
    jacks = [i[1] for i in outs_]
    N = len(jacks)
    return np.sqrt( ((N-1)/N)* np.sum((jacks - np.nanmean(jacks, axis = 0))**2, axis = 0))


def norberg_error_covmat_local(outs_):
    jacks = [i[1] for i in outs_]

    N = len(jacks)
    cov = np.zeros([len(jacks[0]),len(jacks[0])])

    xbar = np.nanmean(jacks, axis = 0)

    for i in np.arange(len(jacks)):

        xi = jacks[i]

        cov += np.outer((xi - xbar),(xi- xbar))

    cov *= (N-1)/N
    
    return cov


def key_data(tab):
    try:
        return tab['RA'], tab['DEC'], tab['Z']
    except KeyError:
        return tab['ra'], tab['dec'], tab['Z']

def zcut_bool(zs, zmin= 0.5, zmax = 1.0):
    return (zs>zmin)*(zs<zmax)


# First make an auto and cross wp function, that takes in the datasets, the redshift limit, etc

def key_data(tab):
    try:
        return tab['RA'], tab['DEC'], tab['Z']
    except KeyError:
        try:
            return tab['ra'], tab['dec'], tab['Z']
        
        except KeyError:
            return tab['RA'], tab['DEC'], tab['best_z']

def zcut_bool(zs, zmin= 0.7, zmax = 1.0):
    return (zs>zmin)*(zs<zmax)

def new_random_zs(ran_ra, target_zs, seed = 1234567):
    zs = np.linspace(0,2, 20)
    y,x, fig = plt.hist(target_zs, density = True, bins = zs);
    plt.close()
    
    ynew = gaussian_filter1d(y, 1)
    xnew = (x[1:] + x[:-1])/2

    full_zs = np.linspace(0,2, 2000)

    y_new = gaussian_filter1d(np.interp(full_zs, xnew, ynew),50)
    full_weights = y_new/simpson(y_new, x = full_zs)
    
    
    np.random.seed(seed)

    random_zs = np.random.choice(full_zs, size=len(ran_ra), p=full_weights/np.sum(full_weights))


    return random_zs

def jackknife_indexes(boxes_r, boxes_d, ra, dec):

    indices = []
    for i in np.arange(len(boxes_r)-1): #Goes RA1, DEC1, DEC2, DEC3... RA2, DEC1, DEC2.... RA3, DEC1, DEC2.....
        for j in np.arange(len(boxes_d)-1):
            in_box_index = np.where( (ra > boxes_r[i]) & (ra < boxes_r[i+1]) & (dec > boxes_d[j]) & (dec < boxes_d[j+1]))
            indices.append(in_box_index)

    list_boxes = np.zeros(len(ra)) 

    for i in range(len(indices)): #Assigns number to each 
        list_boxes[indices[i]] = i 
    

    return list_boxes



def jackknife_boxes(side_number, ra, dec, ramin = -360):
    ind  =(ra>ramin)

    ra = ra[ind]
    dec = dec[ind]

    lenra  = np.max(ra) - np.min(ra)
    lendec = np.max(dec) - np.min(dec)

    box_side_ra  = lenra/float(side_number)
    box_side_dec = lendec/float(side_number)

    boxes_ra  = np.zeros(side_number+1) 
    boxes_dec = np.zeros(side_number+1)

    t = np.linspace(np.deg2rad(np.min(ra)),np.deg2rad(np.max(ra)), 
                            side_number+1)

    p = np.arcsin((np.arange(side_number+1)/side_number) * np.sin(np.deg2rad(np.max(dec)-np.min(dec))))+(np.deg2rad(np.min(dec)))
                                                                                                         

    boxes_ra, boxes_dec = np.rad2deg(t),np.rad2deg(p)

    list_boxes = jackknife_indexes(boxes_ra, boxes_dec, ra, dec)

    return boxes_ra, boxes_dec, list_boxes

def jacknife_mask(randlist, _iter, _box):
    exclude_boxes = randlist[_iter] 

    
    mask = np.isin(_box,exclude_boxes, invert = True)
    print(exclude_boxes)
    return mask



def wp_correlation_cross_i(RA, DEC, CZ,
                         RA2, DEC2, CZ2,
                         rand_RA, rand_DEC, rand_CZ,
                         rand_RA2, rand_DEC2, rand_CZ2,N, N2, rand_N, rand_N2,
                         pimax = 40, nbins = 12, rmin = 0.1, rmax = 40., nthreads = 10):
    # Setup the bins
    bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1) # In Mpc/h
    bin_center = (bins[1:] + bins[:-1])/2
    
    cosmology = 2 #Planck cosmology

    # Auto pair counts in DD
    autocorr=0
    D1D2_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins,
                             RA, DEC, CZ,
                             RA2=RA2, DEC2=DEC2, CZ2=CZ2, verbose = False)
    
#     print('Completed DD pairs')

    # Cross pair counts in DR
    D1R2_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins,
                             RA, DEC, CZ,
                             RA2=rand_RA2, DEC2=rand_DEC2, CZ2=rand_CZ2, verbose = False)

    # Cross pair counts in DR
    D2R1_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins,
                             RA2, DEC2, CZ2,
                             RA2=rand_RA, DEC2=rand_DEC, CZ2=rand_CZ, verbose = False)

#     print('Completed DR pairs')

    # Auto pairs counts in RR
    R1R2_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins,
                             rand_RA, rand_DEC, rand_CZ,
                             RA2=rand_RA2, DEC2=rand_DEC2, CZ2=rand_CZ2, verbose = False)
#     print('Completed RR pairs')

    wp = convert_rp_pi_counts_to_wp(N, N2, rand_N, rand_N2,
                                 D1D2_counts, D1R2_counts,
                                 D2R1_counts, R1R2_counts, nbins, pimax)
    
    return bin_center,  wp



def make_dNdz_specz(speczs, dz = 0.0001, zmin = 0.0, zmax = 3.0):
    z_grid = np.arange(zmin, zmax+dz, dz)
    dndz_total = np.zeros_like(z_grid)

    for i in tqdm(range(len(speczs))):
        _pz = stats.norm.pdf(z_grid, speczs[i], 0.0001)
        norm = _pz/simpson(_pz,x = z_grid)
        if sum(norm) > 0:
            dndz_total += norm

    return z_grid, dndz_total


def desi_dNdz(dataf, zmin, zmax):

    if type(dataf) == str: 
        data = Table.read(dataf)
    
    if type(dataf) == Table: 
        data = dataf

    RA, DEC, Z = key_data(data)
    Z = Z[(Z>(zmin-0.1)) * (Z< (zmax + 0.1))]
    return make_dNdz_specz(Z)

    
def wp_cross_correlation_single(data1f, data2f, random1f, random2f, zmin = 0.7, zmax = 1.0, randmult = 20.,
                        pimax = 40, nbins = 12, rmin = 0.1, rmax = 40., nthreads = 10):
    

    if type(data1f) == str: 
        data1 = Table.read(data1f)
    
    if type(data2f) == str: 
        data2 = Table.read(data2f)

    if type(random1f) == str:
        randoms1 = Table.read(random1f)

    if type(random2f) == str:
        randoms2 = Table.read(random2f)

    if type(data1f) == Table: 
        data1 = data1f
    
    if type(data2f) == Table: 
        data2 = data2f

    if type(random1f) == Table:
        randoms1 = random1f

    if type(random2f) == Table:
        randoms2 = random2f


    RA, DEC, CZ = key_data(data1)
    RA2, DEC2, CZ2 = key_data(data2)

    rand_RA, rand_DEC, rand_CZ = key_data(randoms1)

    rand_RA2, rand_DEC2, rand_CZ2 = key_data(randoms2)
    

    rand_CZ = new_random_zs(rand_RA, CZ, seed = 1234567)
    rand_CZ2 = new_random_zs(rand_RA2, CZ2, seed = 12345678)

    # first a redshift cut:

    zbool =  zcut_bool(CZ, zmin = zmin, zmax=zmax)
    rand_zbool = zcut_bool(rand_CZ, zmin = zmin, zmax=zmax)

    zbool2 =  zcut_bool(CZ2, zmin = zmin, zmax=zmax)
    rand_zbool2 = zcut_bool(rand_CZ2, zmin = zmin, zmax=zmax)


    RA, DEC, CZ = RA[zbool], DEC[zbool], CZ[zbool]
    rand_RA, rand_DEC, rand_CZ = rand_RA[rand_zbool], rand_DEC[rand_zbool], rand_CZ[rand_zbool] 


    RA2, DEC2, CZ2 = RA2[zbool2], DEC2[zbool2], CZ2[zbool2]
    rand_RA2, rand_DEC2, rand_CZ2 = rand_RA2[rand_zbool2], rand_DEC2[rand_zbool2], rand_CZ2[rand_zbool2] 


    choice_bool = np.random.choice(np.arange(len(rand_RA)), size = int(randmult*len(RA)), replace = False)
    choice_bool2 = np.random.choice(np.arange(len(rand_RA2)), size = int(randmult*len(RA2)), replace = False)

    rand_RA, rand_DEC, rand_CZ = rand_RA[choice_bool], rand_DEC[choice_bool], rand_CZ[choice_bool] 
    rand_RA2, rand_DEC2, rand_CZ2 = rand_RA2[choice_bool2], rand_DEC2[choice_bool2], rand_CZ2[choice_bool2] 



    N = len(RA)
    rand_N = len(rand_RA)

    N2 = len(RA2)
    rand_N2 = len(rand_RA2)

    print(N, rand_N,N2, rand_N2 )
    
    bin_center, wp = wp_correlation_cross_i(RA, DEC, CZ,
                         RA2, DEC2, CZ2,
                         rand_RA, rand_DEC, rand_CZ,
                         rand_RA2, rand_DEC2, rand_CZ2,N, N2, rand_N, rand_N2,
                         pimax = pimax, nbins = nbins, rmin = rmin, rmax = rmax, nthreads = nthreads)

    return bin_center, wp


def wp_cross_correlation_jackknife(data1f, data2f, random1f, random2f, zmin = 0.7, zmax = 1.0, randmult = 20.,
                        pimax = 40, nbins = 12, rmin = 0.1, rmax = 40., nthreads = 10, nside_box = 5):
    num_iterations = nside_box**2 


    if type(data1f) == str: 
        data1 = Table.read(data1f)
    
    if type(data2f) == str: 
        data2 = Table.read(data2f)

    if type(random1f) == str:
        randoms1 = Table.read(random1f)

    if type(random2f) == str:
        randoms2 = Table.read(random2f)

    if type(data1f) == Table: 
        data1 = data1f
    
    if type(data2f) == Table: 
        data2 = data2f

    if type(random1f) == Table:
        randoms1 = random1f

    if type(random2f) == Table:
        randoms2 = random2f

    RA, DEC, CZ = key_data(data1)
    RA2, DEC2, CZ2 = key_data(data2)

    rand_RA, rand_DEC, rand_CZ = key_data(randoms1)

    rand_RA2, rand_DEC2, rand_CZ2 = key_data(randoms2)
    

    rand_CZ = new_random_zs(rand_RA, CZ, seed = 1234567)
    rand_CZ2 = new_random_zs(rand_RA2, CZ2, seed = 12345678)

    # first a redshift cut:

    zbool =  zcut_bool(CZ, zmin = zmin, zmax=zmax)
    rand_zbool = zcut_bool(rand_CZ, zmin = zmin, zmax=zmax)

    zbool2 =  zcut_bool(CZ2, zmin = zmin, zmax=zmax)
    rand_zbool2 = zcut_bool(rand_CZ2, zmin = zmin, zmax=zmax)


    RA, DEC, CZ = RA[zbool], DEC[zbool], CZ[zbool]
    rand_RA, rand_DEC, rand_CZ = rand_RA[rand_zbool], rand_DEC[rand_zbool], rand_CZ[rand_zbool] 


    RA2, DEC2, CZ2 = RA2[zbool2], DEC2[zbool2], CZ2[zbool2]
    rand_RA2, rand_DEC2, rand_CZ2 = rand_RA2[rand_zbool2], rand_DEC2[rand_zbool2], rand_CZ2[rand_zbool2] 


    choice_bool = np.random.choice(np.arange(len(rand_RA)), size = int(randmult*len(RA)), replace = False)
    choice_bool2 = np.random.choice(np.arange(len(rand_RA2)), size = int(randmult*len(RA2)), replace = False)

    rand_RA, rand_DEC, rand_CZ = rand_RA[choice_bool], rand_DEC[choice_bool], rand_CZ[choice_bool] 
    rand_RA2, rand_DEC2, rand_CZ2 = rand_RA2[choice_bool2], rand_DEC2[choice_bool2], rand_CZ2[choice_bool2] 
    
    
    boxra, boxdec, _temp = jackknife_boxes(nside_box, RA2, DEC2)

    box_d1 = jackknife_indexes(boxra, boxdec, RA, DEC)
    box_r1 = jackknife_indexes(boxra, boxdec, rand_RA, rand_DEC)
    box_d2 = jackknife_indexes(boxra, boxdec, RA2, DEC2)
    box_r2 = jackknife_indexes(boxra, boxdec, rand_RA2, rand_DEC2)


    random_list_boxes = np.array([[i] for i in np.arange(nside_box**2)])  ##remove only one box at a time 

    

    def main(i):
        
        D1mask = jacknife_mask(random_list_boxes, i, box_d1)
        D2mask = jacknife_mask(random_list_boxes, i, box_d2)
        R1mask = jacknife_mask(random_list_boxes, i, box_r1)
        R2mask = jacknife_mask(random_list_boxes, i, box_r2)
        
        
        RA_j, DEC_j, CZ_j = RA[D1mask], DEC[D1mask], CZ[D1mask]
        rand_RA_j, rand_DEC_j, rand_CZ_j = rand_RA[R1mask], rand_DEC[R1mask], rand_CZ[R1mask] 


        RA2_j, DEC2_j, CZ2_j = RA2[D2mask], DEC2[D2mask], CZ2[D2mask]
        rand_RA2_j, rand_DEC2_j, rand_CZ2_j = rand_RA2[R2mask], rand_DEC2[R2mask], rand_CZ2[R2mask] 


        
        N_j = len(RA_j)
        rand_N_j = len(rand_RA_j)

        N2_j = len(RA2_j)
        rand_N2_j = len(rand_RA2_j)


        out = wp_correlation_cross_i(RA_j, DEC_j, CZ_j,
                             RA2_j, DEC2_j, CZ2_j,
                             rand_RA_j, rand_DEC_j, rand_CZ_j,
                             rand_RA2_j, rand_DEC2_j, rand_CZ2_j,N_j, N2_j, rand_N_j, rand_N2_j,
                             pimax = pimax, nbins = nbins, rmin = rmin, rmax = rmax, nthreads = nthreads)


            
        return out 


    n_jacknife = num_iterations#6
    stacked_dict=[]
    for i in np.arange((n_jacknife)):
        out = main(i)
        stacked_dict.append(out)
    
    return stacked_dict

        

def xir_analytic(r, g,A):
    r0g = A * (gamma(g/2.))/(gamma(1/2) * gamma((g-1)/2.) )
    return r0g/(r**g)


def wprp_integrand_analytic(r, rp, g, A):
    return 2*(xir_analytic(r,g,A) * r)/(np.sqrt(r**2 - rp**2))

# wprp_integrand = lambda r, rp, a: 2*(xir(r,a) * r)/(np.sqrt(r**2 - rp**2))

def wprp_model_analytic_inte(rps, g, A):
    model = np.zeros(len(rps))
    
    for i in tqdm(np.arange(len(rps))):

        rpmin = rps[i]

        rs = np.logspace(np.log10(rpmin),6, 10000000)[1:] #integrate from rp to ~infinity, 
                                        #have checked that this is large enough to converge

        def wprp_integrand_simpson(r):
            return wprp_integrand_analytic(r, rpmin, g, A)


        model[i] = simpson(wprp_integrand_simpson(rs), x = rs)

    return model






def Mmin_from_bias_T10(zgrid, dNdz, bias):

    #Mass range
    M_min = 4e8
    # M_min = 2e10

    M_max = 1e16

    Ms = np.geomspace(M_min, M_max, 10000)


    #Define critical density for spherical collapse
    delta_c = 1.686

    #Define time evolution
    # a_s = np.linspace(1, 0.5, 100)

    hmf = ccl.halos.MassFuncTinker10(cosmo)

        
    # Define and downsample zgrid and dN/dz
    
    z_downsample= np.linspace(0,3,300)
    
    zs = zgrid

    dNdz = dNdz/simps(dNdz, zs)
    
    
    fdNdz = interpolate.interp1d(zs, dNdz)
    
    zs = np.copy(z_downsample)
    
    dNdz = fdNdz(zs)/ simps(fdNdz(zs), zs)
    
    a_s = 1./(1+zs)


    nuvec = np.zeros(shape=(len(Ms), len(a_s)))
    for i, scale in enumerate(a_s):

        nuvec[:,i] = delta_c/ccl.sigmaM(cosmo, Ms, scale)

    ###Mass function



    nmvec = []

    for scale in a_s:
        nmvec.append(hmf.get_mass_function(cosmo, Ms, scale))

    nmvec = np.array(nmvec).T
    nuvec = nuvec

    ##Sheth Tormen mass function parameters

    #Tinker 2010 parameters for \Delta = 200

    Delta = 200
    y = np.log10(Delta)

    A = 1.0 + (0.24*y*np.exp(- ((4/y)**4)) )

    a_ = (0.44*y) - 0.88

    B  = 0.183

    b_ = 1.5

    C = 0.019 + (0.107*y) + (0.19*np.exp(- ((4/y)**4)) )

    c_ = 2.4

    b1vec = 1 - (A*(nuvec**a_)/ ((nuvec**a_) + (delta_c**a_) ) ) + (B*(nuvec**b_)) +  (C*(nuvec**c_))   

    #Sheth Tormen lagrangian bias given a fluctuation nu

    #b1vec = 1+ 1./delta_c * ( a*nuvec**2 - 1 + 2 * p / (1 + (a*nuvec**2)**p))

    # b1vec = 1./delta_c * ( nuvec**2 - 1)

    #Convert to Eulerian bias




    # b2vec = 1./delta_c**2 * (a**2 * nuvec**4 - 3*a*nuvec**2 + 2 * p * (2 * a * nuvec**2 + 2 * p - 1)/((1 + a*nuvec**2)**p))
    # b2vec = b2vec[:,0]

    Mmingrid = np.linspace(np.log10(1e9), 13.5, 300)

    # b1vec = b1vec[:,0]

    def get_av_vals(Mmingrid):



        Mgrid = np.tile(Ms, len(Mmingrid)).reshape(len(Mmingrid), len(Ms))    

    #     idxgrid = np.log10(Ms) > Mmingrid

        nbarvec = np.zeros(shape=(len(Mmingrid)))
        b1arr = np.zeros(shape=(len(Mmingrid)))
        Mbarvec = np.zeros(shape=(len(Mmingrid)))
        logMbarvec = np.zeros(shape=(len(Mmingrid)))


        print(b1vec.shape, nmvec.shape)
        b1vecnmarr = np.einsum('ij, ij->ij', b1vec, nmvec)
        Mbarvecnmarr = np.einsum('i, ij->ij', Ms, nmvec)



        for c, Mmin in tqdm(enumerate(Mmingrid)):
            idxM = np.log10(Ms) > Mmin

    #         print(len)
            nbar = simps(simps(nmvec[idxM], np.log10(Ms[idxM]), axis=0) * dNdz, zs)
            #print(nbar)

            b1 = simps((simps((b1vecnmarr)[idxM], np.log10(Ms[idxM]), axis=0)/nbar)* dNdz, zs)
    #         print(b1.shape)
            Mbar = simps((simps((Mbarvecnmarr)[idxM], np.log10(Ms[idxM]), axis=0)/nbar) * dNdz, zs)



            nbarvec[c] = nbar
            b1arr[c] = b1
            Mbarvec[c] = Mbar

        return nbarvec, b1arr, Mbarvec
    
    nbars, b1s, mbars = get_av_vals(Mmingrid)
    
    x = b1s
    y = Mmingrid
    
    f = interpolate.interp1d(x,y)
    
    x = b1s
    y = mbars
    
    f2 = interpolate.interp1d(x,y)
    
    included_bias_ind = np.where((bias>np.min(b1s)) * (bias<np.max(b1s)))


    bias= bias[included_bias_ind] 

    Mmins = np.array([f(i) for i in bias ])

    Maves = np.array([f2(i) for i in bias ])

    return  Mmins, Maves


def Meff_from_bias_T10(zgrid, dNdz, bias):
    

    #Mass range
    M_min = 4e8
    # M_min = 2e10

    M_max = 1e16

    Ms = np.geomspace(M_min, M_max, 10000)


    #Define critical density for spherical collapse
    delta_c = 1.686

    #Define time evolution
    # a_s = np.linspace(1, 0.5, 100

    # Define and downsample zgrid and dN/dz

    
    z_downsample= np.linspace(0,3,300)
    
    zs = zgrid

    dNdz = dNdz/simps(dNdz, zs)
    
    
    fdNdz = interpolate.interp1d(zs, dNdz)
    
    zs = np.copy(z_downsample)
    
    dNdz = fdNdz(zs)/ simps(fdNdz(zs), zs)

    a_s = 1./(1+zs)


    nuvec = np.zeros(shape=(len(Ms), len(a_s)))
    for i, scale in enumerate(a_s):

        nuvec[:,i] = delta_c/ccl.sigmaM(cosmo, Ms, scale)



    # Tinker 2010 bias(nu)

    Delta = 200
    y = np.log10(Delta)

    A = 1.0 + (0.24*y*np.exp(- ((4/y)**4)) )

    a_ = (0.44*y) - 0.88

    B  = 0.183

    b_ = 1.5

    C = 0.019 + (0.107*y) + (0.19*np.exp(- ((4/y)**4)) )

    c_ = 2.4

    b1vec = 1 - (A*(nuvec**a_)/ ((nuvec**a_) + (delta_c**a_) ) ) + (B*(nuvec**b_)) +  (C*(nuvec**c_))   



    #print(np.shape(b1vec))
    
    b1_dNdz = simps(b1vec * dNdz, zs, axis = 1)
    
    #print(len(b1_dNdz) == len(Ms))
    
    
    x = b1_dNdz
    y = Ms
    
    f = interpolate.interp1d(x,y)
    
    included_bias_ind = np.where((bias>np.min(b1_dNdz)) * (bias<np.max(b1_dNdz)))

    bias= bias[included_bias_ind] 

    Meffs = np.array([f(i) for i in bias ])

    
    return  Meffs
    




def b_z_from_M_T10(zs,logM):

    #will turn log(10)M in to M 
    M = 10**logM
    
    #return b(z) from b(z,M) having given a single Mass and a redshift array to define b(z)
    # M in units of solar masses 
    
    
    #Mass range
    #M = np.logspace(12, 15, 10)

    #Define critical density for spherical collapse
    delta_c = 1.686

    #Define time evolution
    #a_s = np.linspace(1, 0.5, 100)
    
    #zs = 1./a_s - 1
    
    a_s = 1./(1.+zs)
    
    
    nuvec = np.zeros(len(a_s))
    for i, a in enumerate(a_s):

        nuvec[i] = delta_c/ccl.sigmaM(cosmo, M, a) #sigma M needs masses linearly


    nuvec = nuvec

    # Tinker 2010 bias(nu)

    Delta = 200
    y = np.log10(Delta)

    A = 1.0 + (0.24*y*np.exp(- ((4/y)**4)) )

    a_ = (0.44*y) - 0.88

    B  = 0.183

    b_ = 1.5

    C = 0.019 + (0.107*y) + (0.19*np.exp(- ((4/y)**4)) )

    c_ = 2.4

    b1vec = 1 - (A*(nuvec**a_)/ ((nuvec**a_) + (delta_c**a_) ) ) + (B*(nuvec**b_)) +  (C*(nuvec**c_))   


    return b1vec
    

    



def xir(r,a):
    return ccl.correlation_3d(cosmo, a= a, r = r)

def wprp_integrand(r, rp, a):
    return 2*(xir(r,a) * r)/(np.sqrt(r**2 - rp**2))

# wprp_integrand = lambda r, rp, a: 2*(xir(r,a) * r)/(np.sqrt(r**2 - rp**2))

def wprp_model_median_z(rps, z):
    model = np.zeros(len(rps))
    
    a = (1/(1.+z))
    
    for i in tqdm(np.arange(len(rps))):

        rpmin = rps[i]

        rs = np.logspace(np.log10(rpmin),6, 10000000)[1:] #integrate from rp to ~infinity, 
                                        #have checked that this is large enough to converge

        def wprp_integrand_simpson(r):
            return wprp_integrand(r, rpmin, a)

        model[i] = simpson(wprp_integrand_simpson(rs), x = rs)

    return model




def full_calculation_plus_error_dictionary(data1f, data2f, random1f, random2f, zmin = 0.7, zmax = 1.0, randmult = 20.,
                        pimax = 40, nbins = 12, rmin = 0.1, rmax = 40., nthreads = 10,  nside_box = 5):
    
    out = wp_cross_correlation_single(data1f, data2f, random1f, random2f, zmin = zmin, zmax = zmax, randmult = randmult,
                        pimax = pimax, nbins = nbins, rmin = rmin, rmax = rmax, nthreads = nthreads)

    
    outs = wp_cross_correlation_jackknife(data1f, data2f, random1f, random2f, zmin = zmin, zmax = zmax, randmult = randmult,
                        pimax = pimax, nbins = nbins, rmin = rmin, rmax = rmax, nthreads = nthreads, nside_box = nside_box)
    jacks = [i[1] for i in outs]

    error = norberg_error_local(outs)
    covmat = norberg_error_covmat_local(outs)
    
    zgrid, dNdz_temp= desi_dNdz(data1f, zmin, zmax)
    #zgrid2, dNdz2_temp= desi_dNdz(data2f)
    
    dNdz = np.zeros_like(zgrid)
    #dNdz2 = np.zeros_like(zgrid)
    
    dNdz[(zgrid > zmin)*(zgrid < zmax)] = dNdz_temp[(zgrid > zmin)*(zgrid < zmax)]
    #dNdz2[(zgrid > zmin)*(zgrid < zmax)] = dNdz2_temp[(zgrid > zmin)*(zgrid < zmax)] # so far have not needed this, and takes more time than is worth

    median_z = np.average(zgrid, weights= dNdz)
    
    model = wprp_model_median_z(out[0]/little_h, median_z)*little_h # in Mpc/h
    
    # Setup the bins
    bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)  # in Mpc/h
    bin_center = (bins[1:] + bins[:-1])/2  # in Mpc/h
    
    
    dict_ = {
        'rp': out[0],
        'wp': out[1],
        'bins': bins,
        'error': error,
        'covmat': covmat,
        'zgrid': zgrid,
        'dNdz1': dNdz,
        #'dNdz2': dNdz2,
        'z': median_z,
        'model': model, 
        'all_jacks': jacks
    }

    return dict_ #outputting in Mpc/h units


def mult_fit_bias_val_info(x,y,theo, sig = None):
    def func_mult(x, m):
        return  x**0 * (theo*m )

    
    pars, cov = curve_fit(f=func_mult, xdata=x, ydata=y, sigma= sig, p0=[4.],
    	maxfev=10000, absolute_sigma = True, method='lm')

    return pars

def mult_fit_bias_val(x,y,theo, sig = None):
    def func_mult(x, m):
        return  x**0 * (theo*m )

    
    pars, cov = curve_fit(f=func_mult, xdata=x, ydata=y, sigma= sig, p0=[4.],
    	maxfev=10000, absolute_sigma = True, method='lm')

    return pars[0]



def mult_fit_bias_val_covmat(x,y,theo, sig = None):
    def func_mult(x, m):
        return  x**0 * (theo*m )

    
    out= curve_fit(f=func_mult, xdata=x, ydata=y, sigma= sig, p0=[4.],
    	maxfev=10000, absolute_sigma = False, method='lm', full_output = True, nan_policy = 'omit')
    #print(out)
    return out[0][0], np.sqrt(np.diag(out[1]))[0]





def fits_dist_output(fits):
    medi = np.median(fits)
    q84 = np.quantile(fits, .84)
    q16 = np.quantile(fits, .16)
    return medi,q84-medi,medi-q16


def bias_fitting_MC(data_dict, inds, N = 1000, errorkey = 'error'):
    bins = data_dict['bins']

    def get_random_draw_x(ind):
        return np.random.uniform(low = bins[ind] ,high = bins[ind+1], size = N)

    def get_random_draw_y(ind):
        return np.random.normal(loc = data_dict['wp'][ind] ,scale = data_dict[errorkey][ind], size = N)


    xs = np.array([get_random_draw_x(i) for i in inds]).T
    ys = np.array([get_random_draw_y(i) for i in inds]).T

    
    models = np.array([np.interp(i, data_dict['rp'], data_dict['model']) for i in xs])

    fits = np.array([mult_fit_bias_val(xs[i], ys[i], models[i], sig = data_dict[errorkey][inds]) for i in tqdm(np.arange(N))])
    
    data_dict['fits'] = fits
    data_dict['b2_fit'] = fits_dist_output(fits)
    
    return data_dict



def bias_fitting(data_dict, inds, N = 10000, errorkey = 'error'): #now just doing the linear minimization from the covariance



    covmat = data_dict['covmat'][inds[0]:inds[-1]+1, inds[0]:inds[-1]+1]
    inverse = np.linalg.inv(covmat)

    theory = data_dict['model'][inds]
    rp = data_dict['rp'][inds]
    wprp = data_dict['wp'][inds]



    fit, error= mult_fit_bias_val_covmat(rp, wprp, theory, sig = covmat)

    r = wprp - (theory*fit)
    chi2 =  np.matmul(np.matmul(r.T, inverse), r)
    chi2_diagonal = np.matmul(np.matmul(r.T, np.linalg.inv(np.diag(np.diag(covmat)))), r)

    biases = np.linspace(0,6, 100000)

    function = stats.norm.pdf(biases, fit, error)


    fits = np.random.choice(biases, size=N, p=function/np.sum(function))


    data_dict['strict_fit'] = fit
    data_dict['strict_fit_sigma'] = error
    data_dict['fits'] = fits
    data_dict['b2_fit'] = fits_dist_output(fits)
    data_dict['chi2'] = chi2
    data_dict['chi2_diagonal'] = chi2_diagonal
    
    return data_dict

def fix_masses_to_Tinker2010_local(data_dict, bgal):
    dictionary = data_dict
    
    dNdz = dictionary['dNdz1']
    zgrid = dictionary['zgrid']
    biases2 = dictionary['fits']
    biases = biases2/bgal 
    
    Meffs = Meff_from_bias_T10(zgrid, dNdz, biases)
    Mmins, Mbars = Mmin_from_bias_T10(zgrid, dNdz, biases)
    
    dictionary['Meffs'] = Meffs # in Msun
    dictionary['Mmin_dist'] = Meffs # in log10 Msun
    dictionary['Mave_dist'] = Mbars # in Msun
    dictionary['Mave'] = fits_dist_output( Mbars) # in Msun
    #print(dictionary['Mave'])
    return dictionary



def final_wprp_dictionary(data1f, data2f, random1f, random2f, zmin, zmax, fit_inds, randmult = 20.,
                        pimax = 60, nbins = 12, rmin = 0.1, rmax = 40., nthreads = 10,  nside_box = 5, bgal = None, 
                        save = False, savedir = './'):

    temp_dict = full_calculation_plus_error_dictionary( data1f, data2f, random1f, random2f, zmin = zmin, zmax = zmax, randmult = randmult,
                        pimax = pimax, nbins = nbins, rmin = rmin, rmax = rmax, nthreads = nthreads,  nside_box = nside_box)
    
    out = bias_fitting(temp_dict, fit_inds)
    out['dof'] = len(fit_inds) - 1  # Number of data points being fit -1 fitting parameter

    if data1f != data2f:
        if bgal == None:
            print('If you are calculating a cross correlation, provide the bgal of the tracer. Halo masses were not calculated.')
        else:
            out= fix_masses_to_Tinker2010_local(out, bgal)

    return out


# bins = np.logspace(np.log10(-1), np.log10(40), 12 + 1)

# fit_inds = np.arange(len(bins))[5:-1] #1.2 Mpc/h onwards
# fit_inds 



def error_prop_auto(dist):
    _84b2 = np.array([np.quantile(dist[i], .84) - np.quantile(dist[i], .5) for i in np.arange(len(dist))])
    _16b2 = np.array([np.abs(np.quantile(dist[i], .16) - np.quantile(dist[i], .5)) for i in np.arange(len(dist))])
    _50b2 = np.array([np.quantile(dist[i], .5) for i in np.arange(len(dist))])
    
    _50b = np.sqrt(_50b2)
    
    sigma_b_upper = np.array([0.5* _84b2[i]/_50b[i] for i in np.arange(len(dist))])
    sigma_b_lower = np.array([0.5* _16b2[i]/_50b[i] for i in np.arange(len(dist))])
    
    out = np.array([_50b, sigma_b_upper,sigma_b_lower]).T 
    return out

def error_prop_cross(dist, auto_bias):
    
    bg, sbgupper, sbglower = auto_bias.T
    
    _84bb = np.array([np.quantile(dist[i], .84) - np.quantile(dist[i], .5) for i in np.arange(len(dist))])
    _16bb = np.array([np.abs(np.quantile(dist[i], .16) - np.quantile(dist[i], .5)) for i in np.arange(len(dist))])
    _50bb = np.array([np.quantile(dist[i], .5) for i in np.arange(len(dist))])
    
    
    _50bQ =np.array([_50bb[i]/bg[i] for i in np.arange(len(dist))])
    
    sigmabQ_upper = np.array([(_50bQ[i] * np.sqrt((_84bb[i]/_50bb[i])**2 + (sbgupper[i]/bg[i])**2)) for i in np.arange(len(dist))])
    sigmabQ_lower = np.array([(_50bQ[i] * np.sqrt((_16bb[i]/_50bb[i])**2 + (sbglower[i]/bg[i])**2)) for i in np.arange(len(dist))])

    out = np.array([_50bQ, np.abs(sigmabQ_upper),np.abs(sigmabQ_lower)]).T 
    return out


def get_dists_of_interest(of_interest):
    
    #of_interest = '*cross*_1_*1pt2*'
    files = np.sort(glob(of_interest))
    dists_all = np.array([load_dict(i)['dist']for i in files])

    return dists_all



def get_dists_of_interest_local(dicts):
    
    #of_interest = '*cross*_1_*1pt2*'
    dists_all = np.array([i['fits']for i in dicts])

    return dists_all


def full_dist_values_auto(dist):
    full_dist = np.array(dist).flatten()
    _50b2 = np.quantile(full_dist, .5)
    _84b2 = np.quantile(full_dist, .84) - np.quantile(full_dist, .5)
    _16b2 = np.quantile(full_dist, .5) - np.quantile(full_dist, .16)
    
    _50b = np.sqrt(_50b2)
    
    sigma_b_upper = np.array(0.5* _84b2/_50b)
    sigma_b_lower = np.array(0.5* _16b2/_50b)
    
    out = np.array([_50b, sigma_b_upper,sigma_b_lower])
    return out


def full_dist_values_cross(dist, auto_bias):
    
    bg, sbgupper, sbglower = auto_bias.T

        
    full_dist = np.array(dist).flatten()
    _50bb = np.quantile(full_dist, .5)
    _84bb = np.quantile(full_dist, .84) - np.quantile(full_dist, .5)
    _16bb = np.quantile(full_dist, .5) - np.quantile(full_dist, .16)
    
    _50bQ =_50bb/bg
    
    sigmabQ_upper = np.array((_50bQ * np.sqrt((_84bb/_50bb)**2 + (sbgupper/bg)**2)))
    sigmabQ_lower = np.array((_50bQ * np.sqrt((_16bb/_50bb)**2 + (sbgupper/bg)**2)))
    
    
    out = np.array([_50bQ, np.abs(sigmabQ_upper),np.abs(sigmabQ_lower)])
    return out

def full_dist_values(mass_errs):
    
    masses = np.array(mass_errs).T[0]
    errupper = np.array(mass_errs).T[1]
    errlower = np.array(mass_errs).T[2]
    err = np.nanmean([errupper, errlower], axis = 0)
    
    _median = np.nansum(masses/err**2)/np.sum(1./err**2)
    _upper = np.sqrt(1./np.nansum(1./errupper**2))
    _lower = np.sqrt(1./np.nansum(1./errlower**2))
    dists_all = np.array([_median, _upper, _lower])

    return dists_all

def compact_bQ_value_calc(direcinfo, bgal, print_ = False):
    if print_ == True:
        print(np.sort(glob(direcinfo)))
        
    bq = error_prop_cross(get_dists_of_interest(direcinfo), bgal)
    bq_all = full_dist_values(bq)
    
    return bq, bq_all



def compact_bQ_value_calc_local(dicts, bgal, print_ = False):

        
    bq = error_prop_cross(get_dists_of_interest_local(dicts), bgal)
    bq_all = full_dist_values(bq)
    
    return bq, bq_all



def compact_bG_value_calc_local(dicts, print_ = False):

    b = error_prop_auto(get_dists_of_interest_local(dicts))
    b_all = full_dist_values(b)
    
    return b, b_all





def plot_bias_vals(vals, c= 'k', xfactor = 0., label = None, ax = None, marker = 'o', ms = 10, oc = None, ow = 1):
    pos = np.arange(len(vals))

    if ax == None:
        fig, ax = plt.subplots(figsize = (6,6))
        fig.canvas.draw();

    #vals = median_84_16_from_dist(vals)
    for i in np.arange(len(vals))[1:]:
        ax.errorbar(i+xfactor, vals[i][0],yerr = np.array([[vals[i][2]],[vals[i][1]]]), fmt = marker, c = c, markersize = ms, markeredgecolor = oc, markeredgewidth = ow)
        
    ax.errorbar(0+xfactor, vals[0][0],yerr = np.array([[vals[0][2]],[vals[0][1]]]), fmt = marker, c = c, label = label, markersize = ms, markeredgecolor = oc, markeredgewidth = ow)

    
    
def plot_bias_vals_1zbins(vals,names, vals_all = None,median = 0, c= 'k', ax = None, ylabel = None, xfactor = 0., 
                          label = None, title = None, ylim =None, marker = 'o', ms = 10, oc = None, ow = 1):
    
    if oc == 'None':
        oc = c
    
    pos = np.arange(len(vals))
    
    if ax == None:
        fig, ax = plt.subplots(figsize = (6,6))
        fig.canvas.draw();
        
    plot_bias_vals(vals, c = c, xfactor= xfactor, label = label, ax = ax, marker = marker, ms = ms, oc = oc, ow = ow)

    if median == 1:
        _50,_84,_16 =  vals_all


        #ax.axhline(_50, ls = '--', c = c, 
        #       label = r'$\langle b \rangle$ = ${:0.2f}^{{+ {:0.2f} }}_{{- {:0.2f} }}$'.format(_50, _84, _16))
        ax.axhline(_50, ls = '--', c = c, 
               label = r'$\langle b \rangle$ = ${:0.2f}\pm {:0.2f} $'.format(_50, _16))

        #ax.axhline(_50-_16, ls = ':', c = c, label = '_Hidden')

        #ax.axhline(_50+_84, ls = ':', c = c, label = '_Hidden')
        ax.fill_between(np.linspace(-1,8), _50+_84, _50-_16, alpha = 0.1, color = c)

    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel);
    ax.set_xticks(pos.astype(float));
    ax.set_xticklabels(names);
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize = 18);
    ax.set_title(title, fontsize = 23)
    ax.legend(fontsize = 18)
    
    return ax



plt.rc('xtick', labelsize=22, direction='in', top = True)

head_dir = '../production_FINAL_OUTPUTS_L6_20240320_UMAP1_condensed_products_T10_masses/'

def get_masses(of_interest, log = True, lit_h = True, _little_h = little_h):
    
    #of_interest = '*cross*_1_*1pt2*'
    files = np.sort(glob(head_dir+of_interest))

    dists_all = np.array([load_dict(i)['Mave']for i in files])
    if lit_h == True:
        dists_all = dists_all *_little_h

    if log == True:
    # Turn into log

        _lower = np.log10(dists_all.T[0])  - np.log10(dists_all.T[0] - dists_all.T[2]) 

        _upper = np.log10(dists_all.T[0] + dists_all.T[1]) - np.log10(dists_all.T[0])  
        
        _median = np.log10(dists_all.T[0])
        
        dists_all = np.array([_median, _upper, _lower]).T

    return dists_all

def get_masses_dist(of_interest, log = True, lit_h = True, _little_h = little_h):
    
    #of_interest = '*cross*_1_*1pt2*'
    files = np.sort(glob(of_interest))#np.sort(glob(head_dir+of_interest))

    dists_all = np.array([fits_dist_output(load_dict(i)['Mave_dist'])for i in files])
    
    if lit_h == True:
        dists_all = dists_all *_little_h

    if log == True:
    # Turn into log

        _lower = np.log10(dists_all.T[0])  - np.log10(dists_all.T[0] - dists_all.T[2]) 

        _upper = np.log10(dists_all.T[0] + dists_all.T[1]) - np.log10(dists_all.T[0])  
        
        _median = np.log10(dists_all.T[0])
        
        dists_all = np.array([_median, _upper, _lower]).T

    return dists_all



def get_masses_local(dicts, log = True, lit_h = True, _little_h = little_h):
    
    #of_interest = '*cross*_1_*1pt2*'
    files = dicts

    dists_all = np.array([i['Mave']for i in files])
    if lit_h == True:
        dists_all = dists_all *_little_h

    if log == True:
    # Turn into log

        _lower = np.log10(dists_all.T[0])  - np.log10(dists_all.T[0] - dists_all.T[2]) 

        _upper = np.log10(dists_all.T[0] + dists_all.T[1]) - np.log10(dists_all.T[0])  
        
        _median = np.log10(dists_all.T[0])
        
        dists_all = np.array([_median, _upper, _lower]).T

    return dists_all

def get_masses_dist_local(dicts, log = True, lit_h = True, _little_h = little_h):
    
    files = dicts

    dists_all = np.array([fits_dist_output(i['Mave_dist'])for i in files])
    
    if lit_h == True:
        dists_all = dists_all *_little_h

    if log == True:
    # Turn into log

        _lower = np.log10(dists_all.T[0])  - np.log10(dists_all.T[0] - dists_all.T[2]) 

        _upper = np.log10(dists_all.T[0] + dists_all.T[1]) - np.log10(dists_all.T[0])  
        
        _median = np.log10(dists_all.T[0])
        
        dists_all = np.array([_median, _upper, _lower]).T

    return dists_all


def full_dist_values(mass_errs):
    
    masses = np.array(mass_errs).T[0]
    errupper = np.array(mass_errs).T[1]
    errlower = np.array(mass_errs).T[2]
    err = np.nanmean([errupper, errlower], axis = 0)
    
    _median = np.nansum(masses/err**2)/np.sum(1./err**2)
    _upper = np.sqrt(1./np.nansum(1./errupper**2))
    _lower = np.sqrt(1./np.nansum(1./errlower**2))
    dists_all = np.array([_median, _upper, _lower])

    return dists_all

def full_dist_values_masses_prev(of_interest, log = True,lit_h = True, _little_h = little_h):
    files = np.sort(glob(head_dir+of_interest))

    dist = np.array([load_dict(i)['Mave_dist']for i in files])
    
    full_dist = np.array(dist).flatten()
    _50 = np.quantile(full_dist, .5)
    _84 = np.quantile(full_dist, .84) - np.quantile(full_dist, .5)
    _16 = np.quantile(full_dist, .5) - np.quantile(full_dist, .16)
    
    dists_all = np.array([_50, _84,_16])
    
    if lit_h == True:
        _50 = _50 *_little_h
        _84 = _84 *_little_h
        _16 = _16 *_little_h


    
    if log == True:
    # Turn into log

        _lower = np.log10(_50)  - np.log10(_50 - _16) 

        _upper = np.log10(_50 + _84) - np.log10(_50)  
        
        _median = np.log10(_50)
        
        dists_all = np.array([_median, _upper, _lower])

    return dists_all

def plot_bias_vals_1zbins_mass(vals,names, vals_all = None,median = 0, c= 'k', ax = None, ylabel = None, xfactor = 0., label = None, title = None, ylim =None, marker = 'o',ms = 10):
    
    pos = np.arange(len(vals))
    
    if ax == None:
        fig, ax = plt.subplots(figsize = (6,6))
        fig.canvas.draw();
        
    plot_bias_vals(vals, c = c, xfactor= xfactor, label = label, ax = ax, marker = marker, ms = ms)

    if median == 1:
        _50,_84,_16 =  vals_all


        ax.axhline(_50, ls = '--', c = c, 
               label = r'$\langle \log \,M_h \rangle$ = ${:0.2f}^{{+ {:0.2f} }}_{{- {:0.2f} }}$'.format(_50, _84, _16))

        #ax.axhline(_50-_16, ls = ':', c = c, label = '_Hidden')

        #ax.axhline(_50+_84, ls = ':', c = c, label = '_Hidden')
        ax.fill_between(np.linspace(-1,8), _50+_84, _50-_16, alpha = 0.1, color = c)

    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel);
    ax.set_xticks(pos.astype(float));
    ax.set_xticklabels(names);
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize = 18);
    ax.set_title(title, fontsize = 23)
    ax.legend(fontsize = 18)
    
    return ax



def fit_for_all_value(mass_errs):
    
    masses = np.array(mass_errs).T[0]
    errupper = np.array(mass_errs).T[1]
    errlower = np.array(mass_errs).T[2]
    err = np.average([errupper, errlower], axis = 0)
    

    def mult_fit(x,y, sig = None):
        def func_mult(x, m):
            return  x**0 * m 


        pars, cov = curve_fit(f=func_mult, xdata=x, ydata=y, sigma= sig, p0=[4.],
            maxfev=10000, absolute_sigma = True, method='lm')

        # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
        stdevs = np.sqrt(np.diag(cov))
        # Calculate the residuals
        res = y - func_mult(x, *pars)

        return pars, stdevs, res
    
    x= np.arange(len(masses))
    print(x)
    pars, stdevs, res = mult_fit(x,masses, sig = err)
    
    #dists_all = np.array([_median, _upper, _lower])

    return  pars, stdevs

    




