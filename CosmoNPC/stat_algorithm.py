import numpy as np
from mpi4py import MPI
import logging
# from nbodykit.lab import *
from .math_funcs import *
import gc



def calculate_power_spectrum_survey(rfield, stat_attrs, comm, **kwargs):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Extract mesh attributes
    poles = stat_attrs['poles']
    boxsize, nmesh = np.array(stat_attrs['boxsize']), np.array(stat_attrs['nmesh'])
    boxcenter = np.array(stat_attrs['boxcenter'])
    k_min, k_max, k_bins = stat_attrs['k_min'], stat_attrs['k_max'], stat_attrs['k_bins']
    sampler, interlaced = stat_attrs['sampler'], stat_attrs['interlaced']
    I_norm, N0 =  stat_attrs['I_rand'], stat_attrs['N0']

    # Define some useful variables
    k_edge = np.linspace(k_min, k_max, k_bins + 1)
    if rank == 0:
        logging.info(f"Rank {rank}: k_edge = {k_edge}")

    comm.Barrier()
    # Calculate the Fourier transform of the density field
    cfield = rfield.r2c()

    # Compensate the cfield depending on the type of mesh
    compensation = get_compensation(interlaced, sampler)
    cfield.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
    cfield[:] *= boxsize.prod() 
    # very interesting, if the normalization is not done here, 
    # the result will be wrong, 
    if rank == 0:
        logging.info(f"{compensation[0][1].__name__} applied to the density field")

    # Validate the poles 
    validate_poles(poles)

    # Get the kgrid, knorm, and x_grid for binning and spherical harmonics
    if poles == [0]:  # For ell = 0 only, we do not need to calculate the kgrid
        knorm = np.sqrt(sum(np.real(kk)**2 for kk in cfield.x))
        kgrid = None
    else:
        kgrid, knorm = get_kgrid(cfield)
        # kgrid, knorm = get_kgrid_new(cfield) # it will be faster but need further validation
        xgrid = get_xgrid(rfield, boxcenter, boxsize, nmesh)


    results = {} if rank == 0 else None

    # get k_eff and k_num in one particular k_bin
    sub_count, sub_knorm_sum = get_kbin_count(k_bins, k_edge, knorm)
    # gather the results from all ranks
    k_num = comm.reduce(sub_count, op=MPI.SUM, root=0)
    total_knorm_sum = comm.reduce(sub_knorm_sum, op=MPI.SUM, root=0)
    if rank == 0:
        k_eff = total_knorm_sum / k_num
        logging.info(f"Rank {rank}: k_eff = {k_eff}")
        logging.info(f"Rank {rank}: k_num = {k_num}")
    results = {'k_eff': k_eff, "k_num":k_num} if rank == 0 else None

    
    # Loop over the poles
    for pole in poles:
        if rank == 0:
            logging.info(f"Rank {rank}: Processing pole {pole}")

        if pole == 0:
            P_ell_field = np.real(cfield[:])**2 + np.imag(cfield[:])**2 
        else:
            G_ell = get_G_ell(rfield, pole, kgrid, xgrid,compensation,boxsize,comm)
            P_ell_field = cfield[:] * np.conj(G_ell[:])

        
        # Radial binning
        sub_sum = radial_binning(P_ell_field, k_bins, k_edge, knorm)

        # Gather the results from all ranks
        total_sum = comm.reduce(sub_sum, op=MPI.SUM, root=0)

        if rank == 0:
            res = total_sum / k_num
            # Add some factors, substract the shot noise and take the complex conjugate for ell > 0
            res *= (2*pole+1)/ I_norm 
            
            if pole == 0:
                res -= N0
                logging.info(f"Rank {rank}: Shot noise subtracted from P0")
            else:
                if pole & 1: # ell = odd
                    res = 2j * np.imag(res)
                else:
                    res = 2 * np.real(res)

            # Store the results
            results.update({
                f'P{pole}': res,
            })
            logging.info(f"Rank {rank}: P{pole} calculated")

    # Free memory
    del cfield, P_ell_field
    gc.collect()
    if rank == 0:
        logging.info(f"Rank {rank}: Finished processing all poles")
        return results
    else:
        return None



def calculate_power_spectrum_box(rfield, stat_attrs, comm, **kwargs):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Extract mesh attributes
    poles = stat_attrs['poles']
    boxsize, nmesh = np.array(stat_attrs['boxsize']), np.array(stat_attrs['nmesh'])
    k_min, k_max, k_bins = stat_attrs['k_min'], stat_attrs['k_max'], stat_attrs['k_bins']
    sampler, interlaced = stat_attrs['sampler'], stat_attrs['interlaced']
    P_shot = stat_attrs['P_shot']
    NZ = stat_attrs['NZ']
    rsd = np.array(stat_attrs['rsd'])

    if rank == 0:
        logging.info(f"Rank {rank}: Using rsd = {rsd}.")

    # Define some useful variables
    k_edge = np.linspace(k_min, k_max, k_bins + 1)
    if rank == 0:
        logging.info(f"Rank {rank}: k_edge = {k_edge}")
    comm.Barrier()

    # Calculate the Fourier transform of the density field
    cfield = rfield.r2c()
    # Compensate the cfield depending on the type of mesh
    compensation = get_compensation(interlaced, sampler)
    cfield.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
    if rank == 0:
        logging.info(f"{compensation[0][1].__name__} applied to the density field")


    
    # Validate the poles
    validate_poles(poles)

    # Get the kgrid, knorm for binning and Legendre polynomials
    if poles == [0]:  # For ell = 0 only, we do not need to calculate the kgrid
        knorm = np.sqrt(sum(np.real(kk)**2 for kk in cfield.x))
        kgrid = None
    else:
        kgrid, knorm = get_kgrid(cfield)

    # clear zero-mode in the Fourier space
    cfield[knorm == 0.] = 0. 
    if rank == 0:
        logging.info(f"Rank {rank}: Zero-mode in Fourier space cleared")


    P_field = np.real(cfield[:])**2 + np.imag(cfield[:])**2


    # save some memory, cfield, rfield are not needed anymore
    del cfield, rfield
    gc.collect()


    results = {} if rank == 0 else None

    # get k_eff and k_num in one particular k_bin
    sub_count, sub_knorm_sum = get_kbin_count(k_bins, k_edge, knorm)
    # gather the results from all ranks
    k_num = comm.reduce(sub_count, op=MPI.SUM, root=0)
    total_knorm_sum = comm.reduce(sub_knorm_sum, op=MPI.SUM, root=0)
    if rank == 0:
        k_eff = total_knorm_sum / k_num
        logging.info(f"Rank {rank}: k_eff = {k_eff}")
        logging.info(f"Rank {rank}: k_num = {k_num}")
    results = {'k_eff': k_eff, "k_num":k_num} if rank == 0 else None

    # Loop over the poles
    for pole in poles:
        if rank == 0:
            logging.info(f"Rank {rank}: Processing pole {pole}")

        if pole & 1: # for ell = odd, the multipoles are 0 strictly
            res = np.zeros(k_bins).astype("float128")
            if rank == 0:
                logging.info(f"Rank {rank}: ell = {pole}, Odd multipoles are automatically set to zero")
        else:
            if pole == 0:
                sub_sum = radial_binning(P_field, k_bins, k_edge, knorm)
            else:
                L_ells = get_legendre(pole, rsd[0], rsd[1], rsd[2])
                if rank == 0:
                    logging.info(f"Rank {rank}: L_ells = {L_ells.expr}")
                sub_sum = radial_binning(P_field * L_ells(kgrid[0],kgrid[1],kgrid[2]), \
                                        k_bins, k_edge, knorm)
            
            # Gather the results from all ranks
            total_sum = comm.reduce(sub_sum, op=MPI.SUM, root=0)

            if rank == 0:
                res = total_sum / k_num
                res *= boxsize.prod() 
                """
                This correction see: 
                https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/algorithms/fftpower.html#FFTPower
                or we can interpret it as, for the fft operation, just like the survey-like case,
                we need correct the boxsize.prod()**2, however,
                similar to the survey case divided by I_norm, we need to divide Volume
                since by definition
                $P(\mathbf{k})=\frac{|\delta(\mathbf{k})|^2}{V}$
                conbined with this 2 corrections, we finally get the boxsize.prod()
                """
                if pole == 0:
                    res -= P_shot
                    logging.info(f"Rank {rank}: Shot noise subtracted from P0")
                else:
                    res *= 2*pole+1

        # Store the results
        if rank == 0:
            results.update({
                f'P{pole}': res,
            })
            logging.info(f"Rank {rank}: P{pole} calculated")

    # Free memory
    del P_field
    gc.collect()
    if rank == 0:
        logging.info(f"Rank {rank}: Finished processing all poles")
        return results
    else:
        return None



def get_G_ell(rfield, ell, kgrid, xgrid,compensation,boxsize,comm):
    """
    Calculate the G_ell function for the given ell.
    This function is a placeholder and should be replaced with the actual implementation.
    \mathcal{G}_\ell(\mathbf{k})= \frac{4\pi}{2\ell+1} \left[\frac{1}{2}F_\ell^0(\mathbf{k})
    +\sum_{m=1}^\ell F_\ell^m(\mathbf{k})\right]
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    Ylms = [get_Ylm(ell, m) for m in range(ell + 1)]
    rf = rfield * Ylms[0](xgrid[0], xgrid[1], xgrid[2])
    G_ell = rf.r2c() 
    G_ell[:] *= Ylms[0](kgrid[0], kgrid[1], kgrid[2])/2

    for m in range(1, ell + 1):
        rf = rfield * np.conj(Ylms[m](xgrid[0], xgrid[1], xgrid[2]))
        cf = rf.r2c()
        cf[:] *= Ylms[m](kgrid[0], kgrid[1], kgrid[2])
        G_ell[:] += cf[:]

    # recollect the memory
    del rf, cf
    gc.collect()

    G_ell.apply(out=Ellipsis, func=compensation[0][1], kind=compensation[0][2])
    if rank == 0:
        logging.info(f"{compensation[0][1].__name__} applied to G_ell")
        # print(f"rank {rank}: type of G_ell = {type(G_ell)}")
    G_ell[:] *= 4*np.pi*boxsize.prod() / (2*ell + 1)

    return G_ell



def validate_poles(poles):
    """
    Validate the poles input.
    Raise ValueError if the input is not valid.
    1. All elements must be non-negative integers.
    2. No duplicate values.
    3. If more than one value, they must be sorted in ascending order.
    """
    if not all(isinstance(p, int) and p >= 0 for p in poles):
        raise ValueError("All elements in 'poles' must be non-negative integers.")
    if len(poles) != len(set(poles)):
        raise ValueError("'poles' contains duplicate values.")
    if len(poles) > 1 and poles != sorted(poles):
        raise ValueError("'poles' must be sorted in ascending order if it contains more than one value.")



def get_kbin_count(k_bins, k_edge, knorm):
    """
    Count the number of points as well as the sum k-distance in each k-bin 
    """
    # Initialize the result array
    sub_count = np.zeros(k_bins).astype("float128")
    sub_knorm_sum = np.zeros(k_bins).astype("float128")

    # Loop over the bins and sum the values in each bin
    for i in range(k_bins):
        mask = np.logical_and(knorm >= k_edge[i], knorm < k_edge[i+1])
        sub_count[i] = np.sum(mask)
        sub_knorm_sum[i] = np.sum(knorm[mask])

    del mask
    gc.collect()

    return sub_count, sub_knorm_sum



def radial_binning(kfield, k_bins, k_edge, knorm):
    """
    Radial binning of the Fourier transform of the density field
    """
    # Initialize the result array
    sub_sum = np.zeros(k_bins).astype(kfield.dtype)

    # Loop over the bins and sum the values in each bin
    for i in range(k_bins):
        mask = np.logical_and(knorm >= k_edge[i], knorm < k_edge[i+1])
        sub_sum[i] = np.sum(kfield[mask])

    del mask
    gc.collect()

    return sub_sum




"""
Taken from nbodykit.algorithms.convpower.catalogmesh
see Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240> for details
"""
def get_compensation(interlaced, sampler):
    """
    Return the compensation function, which corrects for the
    windowing kernel.
    """
    if interlaced:
        d = {'cic' : CompensateCIC,
             'tsc' : CompensateTSC,
             'pcs' : CompensatePCS,
            }
    else:
        d = {'cic' : CompensateCICShotnoise,
             'tsc' : CompensateTSCShotnoise,
             'pcs' : CompensatePCSShotnoise,
            }

    if not sampler in d:
        raise ValueError("compensation for window %s is not defined" % sampler)

    filter = d[sampler]

    return [('complex', filter, "circular")]
