import numpy as np
from pmesh import ParticleMesh
from numpy.lib import format
from mpi4py import MPI
import logging
# from nbodykit.lab import *
from .catalog_processor import npy_reader,fits_reader, add_completeness_weight,catalog_reader
import warnings
import gc


warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_mesh_box(catalogs, correlation_mode, nmesh, geometry, boxsize, 
                 sampler, interlaced, column_names, comm):
    """
    Generate mesh fields and compute number densities for multi-galaxy catalogs in periodic boxes.
    Returns:
        (mesh_attrs, rfield_a) for auto; (mesh_attrs, rfield_a, rfield_b) for cross.
    """
    rank = comm.Get_rank()

    print("*"*20, "rank = ",rank)

    Nmesh = np.array(nmesh)
    BoxSize = np.array(boxsize)

    # Load data catalogs
    data_a = catalog_reader(catalogs["data_a"], geometry, column_names, None, None, None, comm)
    sub_N_gal_a = data_a['WEIGHT'].shape[0]
    sub_weight_sum_a = np.sum(data_a['WEIGHT'])
    N_gal_a = comm.reduce(sub_N_gal_a, op=MPI.SUM, root=0)
    weight_sum_a = comm.reduce(sub_weight_sum_a, op=MPI.SUM, root=0)
    NZ_a = weight_sum_a / np.prod(BoxSize) if rank == 0 else None


    print("*"*50,data_a.shape)

    if correlation_mode == "cross":
        data_b = catalog_reader(catalogs["data_b"], geometry, column_names, None, None, None, comm)
        sub_N_gal_b = data_b['WEIGHT'].shape[0]
        sub_weight_sum_b = np.sum(data_b['WEIGHT'])
        N_gal_b = comm.reduce(sub_N_gal_b, op=MPI.SUM, root=0)
        weight_sum_b = comm.reduce(sub_weight_sum_b, op=MPI.SUM, root=0)
        NZ_b = weight_sum_b / np.prod(BoxSize) if rank == 0 else None

    # Shot noise
    if rank == 0:
        P_shot = 1.0 / NZ_a if correlation_mode == "auto" else  0.0

    
    # Mesh attributes
    mesh_attrs = {}
    if rank == 0:
        if correlation_mode == "auto":
            mesh_attrs.update({'NZ_a': NZ_a, "N_gal_a": N_gal_a})
        else:
            mesh_attrs.update({'NZ_a': NZ_a, 'NZ_b': NZ_b, "N_gal_a": N_gal_a, "N_gal_b": N_gal_b})
        mesh_attrs.update({'interlaced': interlaced, 'sampler': sampler, "P_shot": P_shot})

    # boardcast mesh_attrs to all ranks
    mesh_attrs = comm.bcast(mesh_attrs, root=0)

    # Mesh generation
    pm = ParticleMesh(BoxSize=BoxSize, Nmesh=Nmesh, dtype='complex128', resampler='tsc',comm=comm)
    # note that, the dtype here must be complex, otherwise, the r2c will be wrong


    # decompose positions and paint to mesh
    layout_a = pm.decompose(data_a['Position'])

    rfield_a = pm.paint(data_a['Position'], mass=data_a['WEIGHT'],layout=layout_a)


    if correlation_mode == "cross":
        layout_b = pm.decompose(data_b['Position'])
        rfield_b = pm.paint(data_b['Position'], mass=data_b['WEIGHT'],layout=layout_b)
    else:
        rfield_b = None


    # Memory cleanup
    del data_a
    if correlation_mode == "cross":
        del data_b
    gc.collect()

    # if rank == 0:
    #     print(np.array(rfield_a)[:5,:5,:5])
    #     print(rfield_a.x)
    #     print(rfield_a.i)
    # print("shape of rfield_a:", rfield_a.shape, "in Rank ", rank)
    
    print(f"rank {rank}: sum of rfield_a = {np.sum(np.array(rfield_a))}")

    return mesh_attrs, rfield_a, rfield_b




def get_mesh_pk_survey(catalogs, correlation_mode, nmesh, geometry, column_names, boxsize, sampler, \
                interlaced,z_range, comp_weight_plan, para_cosmo,comm):
    
    """
        Generate a mesh and compute the power spectrum.
        Parameters:
            catalogs (dict): A dictionary containing the paths to the data and random catalogs.
            correlation_mode (str): The correlation mode ("auto" or "cross").
            nmesh (list): The number of mesh points in each dimension.
            boxsize (list): The size of the box in each dimension.
            geometry (str): The geometry type ("box-like" or "survey-like").
            column_names (list): List of column names for position and weight which only works in some cases.
            sampler (str): The type of mesh sampler to use.
            interlaced (bool): Whether to use interlacing.
            z_range (tuple): The redshift range to filter the data and randoms.
            comp_weight_plan (dict): A dictionary containing the completeness weight plan.
            para_cosmo (dict): A dictionary containing cosmological parameters.
            comm (MPI.Comm): The MPI communicator.
        Returns:
            tuple: A tuple containing the mesh and mesh attributes.
    """
    # Initialize MPI
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Store attributes for the mesh
    mesh_attrs = {}

    # Convert nmesh and boxsize to numpy arrays
    Nmesh = np.array(nmesh)
    BoxSize = np.array(boxsize)

    # Load data and random catalogs
    randoms_a = catalog_reader


    # generate data and randoms catalogs with proper columns for further processing
    data,randoms = catalog_reader(catalogs, geometry, column_names, \
                                  z_range, comp_weight_plan, para_cosmo, comm)
 

    # Create the FKP catalog and generate the mesh
    fkp = FKPCatalog(data, randoms)
    mesh = fkp.to_mesh(
        Nmesh=Nmesh,
        nbar='NZ',
        BoxSize=BoxSize,
        fkp_weight='WEIGHT_FKP',
        comp_weight='WEIGHT',
        window=sampler,
        interlaced=interlaced
    )
    
    rfield = mesh.compute()
    logging.info(f"Mesh shape: {rfield.shape} in Rank {rank}")


    mesh_attrs.update({
        'boxcenter': mesh.attrs["BoxCenter"]
    })

    # free some memory
    del fkp, mesh
    gc.collect()

    # prepare for getting the ratio of number of data and randoms, alpha
    sub_weight_sum_data = np.sum(np.array(data['WEIGHT'] * data['WEIGHT_FKP']))
    sub_weight_sum_randoms = np.sum(np.array(randoms['WEIGHT'] * randoms['WEIGHT_FKP']))

    # prepare for getting the normalization, I
    # $I_{\text {ran }}=\alpha \sum_{\text {rand }} n_{\mathrm{g}}^{\prime} w_{\mathrm{c}} w_{\mathrm{FKP}}^2$
    sub_I_sum_data = np.sum(np.array(data['WEIGHT'] * data['NZ'] * \
                                     data['WEIGHT_FKP'] * data['WEIGHT_FKP']))
    sub_I_sum_randoms = np.sum(np.array(randoms['WEIGHT'] * randoms['NZ'] *\
                                         randoms['WEIGHT_FKP'] * randoms['WEIGHT_FKP']))

    # prepare for getting the shot noise, N 
    sub_N_sum_data = np.sum((np.array(data['WEIGHT'] * data['WEIGHT_FKP']))**2)
    sub_N_sum_randoms = np.sum((np.array(randoms['WEIGHT'] * randoms['WEIGHT_FKP']))**2)

    # Gather results from all ranks to the root
    gathered_results = comm.gather(
        [sub_weight_sum_data, sub_weight_sum_randoms, 
         sub_I_sum_data, sub_I_sum_randoms, 
         sub_N_sum_data, sub_N_sum_randoms], 
        root=0
    )

    if rank == 0:
        # Aggregate results from all ranks
        (weight_sum_data, weight_sum_randoms, 
         I_sum_data, I_sum_randoms, 
         N_sum_data, N_sum_randoms) = map(np.sum, zip(*gathered_results))

        # Calculate alpha
        alpha = weight_sum_data / weight_sum_randoms
        logging.info(f"Alpha = {alpha}")

        # Calculate normalization, I
        I_rand = alpha * I_sum_randoms
        I_data = I_sum_data
        logging.info(f"Normalization: I_rand = {I_rand}, I_data = {I_data}")

        # Calculate shot noise, N0
        N0 = (N_sum_data + alpha**2 * N_sum_randoms) / I_rand
        logging.info(f"Shot noise, N0 = {N0}")

        # Store attributes in mesh_attrs
        mesh_attrs.update({
            'alpha': alpha,
            'I_rand': I_rand,
            'I_data': I_data,
            'N0': N0
        })
    else:
        # Initialize mesh_attrs for non-root ranks
        mesh_attrs = None

    # Broadcast mesh_attrs to all ranks
    mesh_attrs = comm.bcast(mesh_attrs, root=0)

    # free the memory
    del data, randoms
    gc.collect()
    return rfield, mesh_attrs
