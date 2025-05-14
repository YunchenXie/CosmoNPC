import numpy as np
from mpi4py import MPI
import logging
import time 

from .mesh_generator import *
from .stat_algorithm import *
import os


def run_task(task_name,geometry,catalogs,para_cosmo,**kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger(__name__)
    if rank == 0:
        logging.info(
            f"Running {task_name} task with catalogs:{catalogs},\n"
            f"para_cosmo:{para_cosmo},\n"
            f"Other parameters:{kwargs}"
        )

    if task_name == "pk":
        time_start = time.time()

        # check if the kmax is larger than the Nyquist frequency
        if 'k_max' in kwargs:
            k_max = kwargs['k_max']
            nmesh = kwargs['nmesh']
            boxsize = kwargs['boxsize']
            nyquist_freq = np.pi * nmesh[0] / boxsize[0]
            logging.info(f"Nyquist frequency: {nyquist_freq}")
            if k_max >= nyquist_freq:
                raise ValueError(f"k_max {k_max} is larger than the Nyquist frequency {nyquist_freq}. Please choose a smaller k_max.")


        if geometry == "survey-like":
            if rank == 0:
                logging.info("Using survey-like geometry...")
            rfield,stat_attrs = get_mesh_pk_survey(catalogs, nmesh=kwargs['nmesh'], 
                        geometry=geometry,
                        column_names=kwargs['column_names'],
                        boxsize=kwargs['boxsize'], 
                        sampler=kwargs['sampler'], 
                        interlaced=kwargs['interlaced'],
                        z_range=kwargs['z_range'],
                        comp_weight_plan = kwargs['comp_weight_plan'],
                        para_cosmo=para_cosmo,  comm=comm)
        else:
            if rank == 0:
                logging.info("Using box geometry...")
            rfield,stat_attrs = get_mesh_pk_box(catalogs, nmesh=kwargs['nmesh'],
                        geometry=geometry, 
                        boxsize=kwargs['boxsize'], 
                        sampler=kwargs['sampler'], 
                        interlaced=kwargs['interlaced'],
                        column_names=kwargs['column_names'],
                        comm=comm)
        
        # add more information e.g. poles, kmin, kmax, nk, etc.
        stat_attrs.update(kwargs)
        # print(stat_attrs)
        
        # make sure all ranks have finished before performing FFTs
        comm.Barrier()
        time_rfield = time.time()
        if rank == 0:
            logging.info(f"Time to create FKP overdensity field: {time_rfield - time_start:.2f} seconds")
        

        # Compute power spectrum
        if geometry == "survey-like":
            pk_res = calculate_power_spectrum_survey(rfield, stat_attrs, comm = comm, **kwargs)
        else:
            pk_res = calculate_power_spectrum_box(rfield, stat_attrs, comm = comm, **kwargs)

        # Save the power spectrum results
        if rank == 0:
            pk_res.update(stat_attrs)
            logging.info(f"Power spectrum result: {pk_res}")
            # save the result to a file
            output_dir = kwargs.get('output_dir')
            catalog_name = os.path.splitext(os.path.basename(catalogs['data_path']))[0]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"pk_res_{catalog_name}.npy")
            np.save(output_path, pk_res)
            logging.info(f"Power spectrum result saved to {output_path}")
        

        time_pk = time.time()
        if rank == 0:
            logging.info(f"Time to compute power spectrum: {time_pk - time_rfield:.2f} seconds")
            logging.info(f"Total time for pk task: {time_pk - time_start:.2f} seconds")
            
    return None