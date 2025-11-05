import numpy as np
from mpi4py import MPI
import logging
import time 

from .mesh_generator import *
from .stat_algorithm import *
import os


def run_task(statistic,correlation_mode, geometry,catalogs,**kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(levelname)s - %(message)s')

    if rank == 0:
        logging.info(
            f"Running {correlation_mode} {statistic} task with catalogs:{catalogs},\n"
        )

    k_max, nmesh, boxsize = kwargs['k_max'], kwargs['nmesh'], kwargs['boxsize']
    nyquist_freq = np.pi * nmesh[0] / boxsize[0]
    if rank == 0:
        logging.info(f"Nyquist frequency: {nyquist_freq}")

    if statistic == "pk":
        time_start = time.time()
        # check if the kmax is larger than the Nyquist frequency
        if k_max >= nyquist_freq:
            raise ValueError(f"k_max {k_max} is larger than the Nyquist frequency {nyquist_freq}. \
                                Please choose a smaller k_max.")

        if geometry == "survey-like":
            if rank == 0:
                logging.info("Using survey-like geometry...")
            rfield,stat_attrs = get_mesh_pk_survey(catalogs, 
                        correlation_mode, 
                        nmesh=nmesh, 
                        geometry=geometry,
                        column_names=kwargs['column_names'],
                        boxsize=boxsize, 
                        sampler=kwargs['sampler'], 
                        interlaced=kwargs['interlaced'],
                        z_range=kwargs['z_range'],
                        comp_weight_plan = kwargs['comp_weight_plan'],
                        para_cosmo=kwargs['para_cosmo'],  comm=comm)
        else:
            if rank == 0:
                logging.info("Using box geometry...")
            # check if all pole in poles are odd, if true, raise an error
            poles = kwargs['poles']
            if all(p % 2 == 1 for p in poles):
                raise ValueError(f"All poles in {poles} are odd. Please choose at least one even pole, \
                                 since odd poles are defaultly set to 0 for periodic boxes.")

            stat_attrs, rfield_a, rfield_b = get_mesh_box(catalogs,
                        correlation_mode,
                        nmesh=kwargs['nmesh'],
                        geometry=geometry, 
                        boxsize=kwargs['boxsize'], 
                        sampler=kwargs['sampler'], 
                        interlaced=kwargs['interlaced'],
                        column_names=kwargs['column_names'],
                        comm=comm)
        
        # add more information
        stat_attrs.update(kwargs)
        stat_attrs['nyquist_freq'] = nyquist_freq

        
        # make sure all ranks have finished before performing FFTs
        comm.Barrier()
        time_rfield = time.time()
        if rank == 0:
            logging.info(f"Time to create FKP overdensity field(s): {time_rfield - time_start:.2f} seconds")
        

        # Compute power spectrum
        if geometry == "survey-like":
            pk_res = calculate_power_spectrum_survey(rfield, stat_attrs, comm = comm, **kwargs)
        else:
            pk_res = calculate_power_spectrum_box(rfield_a, rfield_b, correlation_mode,\
                                                   stat_attrs, comm = comm, **kwargs)

        # Save the power spectrum results
        if rank == 0:
            pk_res.update(stat_attrs)
            logging.info(f"Power spectrum result: {pk_res}")
            # save the result to a file
            output_dir = kwargs.get('output_dir')
            catalog_name = os.path.splitext(os.path.basename(catalogs['data_a']))[0]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"pk_res_{catalog_name}.npy")
            np.save(output_path, pk_res)
            logging.info(f"Power spectrum result saved to {output_path}")
        

        time_pk = time.time()
        if rank == 0:
            logging.info(f"Time to compute power spectrum: {time_pk - time_rfield:.2f} seconds")
            logging.info(f"Total time for pk task: {time_pk - time_start:.2f} seconds")


    # if task_name == "bk_sco":
    #     time_start = time.time()

    #     # check if the kmax is larger than the Nyquist frequency
    #     if 'k_max' in kwargs:
    #         k_max = kwargs['k_max']
    #         nmesh = kwargs['nmesh']
    #         boxsize = kwargs['boxsize']
    #         nyquist_freq = np.pi * nmesh[0] / boxsize[0]
    #         if rank == 0:
    #             logging.info(f"Nyquist frequency: {nyquist_freq}")
    #         if k_max >= nyquist_freq:
    #             raise ValueError(f"k_max {k_max} is larger than the Nyquist frequency {nyquist_freq}. \
    #                              Please choose a smaller k_max.")
        
    #     if geometry == "survey-like":
    #         if rank == 0:
    #             logging.info("Using survey-like geometry...")
    #         rfield,stat_attrs = get_mesh_bk_sco_survey(catalogs, nmesh=kwargs['nmesh'], 
    #                     geometry=geometry,
    #                     column_names=kwargs['column_names'],
    #                     boxsize=kwargs['boxsize'], 
    #                     sampler=kwargs['sampler'], 
    #                     interlaced=kwargs['interlaced'],
    #                     z_range=kwargs['z_range'],
    #                     comp_weight_plan = kwargs['comp_weight_plan'],
    #                     para_cosmo=para_cosmo,  comm=comm)
    #     else:
    #         if rank == 0:
    #             logging.info("Using box geometry...")
    #         poles = kwargs['poles']
    #         if all(p % 2 == 1 for p in poles):
    #             raise ValueError(f"All poles in {poles} are odd. Please choose at least one even pole, \
    #                              since odd poles are defaultly set to 0 for periodic boxes.")
    #         rfield,stat_attrs = get_mesh_box(catalogs, nmesh=kwargs['nmesh'],
    #                     geometry=geometry, 
    #                     boxsize=kwargs['boxsize'], 
    #                     sampler=kwargs['sampler'], 
    #                     interlaced=kwargs['interlaced'],
    #                     column_names=kwargs['column_names'],
    #                     comm=comm)
        
    #     # add more information e.g. poles, kmin, kmax, nk, etc.
    #     stat_attrs.update(kwargs)

    #     # make sure all ranks have finished before performing FFTs
    #     comm.Barrier()
    #     time_rfield = time.time()
    #     if rank == 0:
    #         logging.info(f"Time to create FKP overdensity field: {time_rfield - time_start:.2f} seconds")
        
    #     # compute bispectrum in the form of Scoccimarro estimator
    #     if geometry == "survey-like":
    #         bk_res = calculate_bk_sco_survey(rfield, stat_attrs, comm = comm, **kwargs)
    #     else:
    #         bk_res = calculate_bk_sco_box(rfield, stat_attrs, comm = comm, **kwargs)
    #         pk_res = calculate_power_spectrum_box(rfield, stat_attrs, comm = comm, **kwargs)
    #         if rank == 0:
    #             bk_res.update(pk_res)
    #     # Save the bispectrum results
    #     if rank == 0:
    #         bk_res.update(stat_attrs)
    #         logging.info(f"Bispectrum result: {bk_res}")
    #         # save the result to a file
    #         output_dir = kwargs.get('output_dir')
    #         catalog_name = os.path.splitext(os.path.basename(catalogs['data_path']))[0]
    #         if not os.path.exists(output_dir):
    #             os.makedirs(output_dir)
    #         output_path = os.path.join(output_dir, f"bk_sco_res_{catalog_name}.npy")
    #         np.save(output_path, bk_res)
    #         logging.info(f"Bispectrum result saved to {output_path}")

        
    # return None


