import numpy as np
from numpy.lib import format
from mpi4py import MPI
import logging
from nbodykit.lab import *
import warnings
import gc


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def get_mesh_pk_box(catalogs, nmesh, geometry, boxsize, sampler, \
                interlaced,column_names,comm):
    """
        Generate a mesh and compute the power spectrum for a periodic box catalog.
        Parameters:
            catalogs (dict): A dictionary containing the paths to the data and random catalogs.
            nmesh (list): The number of mesh points in each dimension.
            boxsize (list): The size of the box in each dimension.
            sampler (str): The type of mesh sampler to use.
            interlaced (bool): Whether to use interlacing.
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
    BoxCenter = BoxSize / 2.0

    # Load data and random catalogs
    data = catalog_reader(catalogs,  geometry, column_names, None, None, None, comm)
    # data['Position'] -= BoxCenter * 20

    # compute the sum of the weights and the number of galaxies
    sub_N_gal = np.array(data['WEIGHT']).shape[0]
    sub_weight_sum = np.sum(np.array(data['WEIGHT']))

    # Gather the N_gal and sum of weights from all ranks
    N_gal = comm.reduce(sub_N_gal, op=MPI.SUM, root=0)
    weight_sum = comm.reduce(sub_weight_sum, op=MPI.SUM, root=0)

    if rank == 0:
        NZ = weight_sum / np.prod(BoxSize)
        logging.info(f"Number density (NZ): {NZ}")
        logging.info(f"Total number of data points: {N_gal}")
        mesh_attrs.update({
            'NZ': NZ,            
            'interlaced': interlaced,
            'sampler': sampler,
            "N_gal": N_gal,
            "P_shot": 1.0 / NZ
        })
    else:
        # Initialize NZ and mesh_attrs for non-root ranks
        NZ = None
        mesh_attrs = {}


    # Broadcast the number density and mesh attributes to all ranks
    NZ = comm.bcast(NZ, root=0)
    mesh_attrs = comm.bcast(mesh_attrs, root=0)

    # Generate the mesh
    mesh = data.to_mesh(Nmesh=Nmesh[0],
            BoxSize=BoxSize,
            weight='WEIGHT',
            resampler=sampler,
            interlaced=interlaced,
            # compensated = True,
            dtype='complex128'
            )
    rfield = mesh.compute()

    logging.info(f"Mesh shape: {rfield.shape} in Rank {rank}")



    # Renomalize the rfield and get the overdensity
    # rfield -= 1.0

    # free the memory
    del data, mesh
    gc.collect()

    return rfield, mesh_attrs

    

def get_mesh_pk_survey(catalogs, nmesh,geometry, column_names, boxsize, sampler, \
                interlaced,z_range, comp_weight_plan, para_cosmo,comm):
    
    """
        Generate a mesh and compute the power spectrum.
        Parameters:
            catalogs (dict): A dictionary containing the paths to the data and random catalogs.
            nmesh (list): The number of mesh points in each dimension.
            boxsize (list): The size of the box in each dimension.
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

    # generate data and randoms catalogs with proper columns for further processing
    data,randoms = catalog_reader(catalogs, geometry, column_names, \
                                  z_range, comp_weight_plan, para_cosmo, comm)
 

    # Create the FKP catalog and generate the mesh
    fkp = FKPCatalog(data, randoms)
    mesh = fkp.to_mesh(
        Nmesh=Nmesh[0],
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



def catalog_reader(catalogs, geometry, column_names, z_range, comp_weight_plan, para_cosmo, comm):
    """
    Reads catalogs based on the specified geometry and returns the processed data.

    Parameters:
        catalogs (dict): Paths to the data and random catalogs.
        geometry (str): The geometry type ("box-like" or "survey-like").
        column_names (list): List of column names for position and weight.
        z_range (tuple): Redshift range for filtering (only for survey-like geometry).
        comp_weight_plan (dict): Completeness weight configuration.
        para_cosmo (dict): Cosmological parameters.
        comm (MPI.Comm): MPI communicator.

    Returns:
        ArrayCatalog or tuple: Processed data catalog(s).
    """
    rank = comm.Get_rank()
    supported_types = {"npy", "fits"}  # Supported file types for catalogs

    if geometry == "box-like":
        # Handle box-like geometry
        data_path = catalogs.get("data_path")
        data_ext = data_path.split('.')[-1].lower() if data_path else None

        # Check if the file type is supported
        if data_ext not in supported_types:
            raise ValueError(f"Unsupported data file type: {data_ext}")

        if data_ext == "npy":
            # Read .npy file and process it
            data_arr = npy_reader(data_path, comm)
            position_indices = [column_names.index(axis) for axis in ['x', 'y', 'z']]
            data_cat = ArrayCatalog({"Position": np.column_stack([data_arr[:, idx] for idx in position_indices])})

            # Check for weight column
            if 'w' in column_names:
                data_cat['WEIGHT'] = data_arr[:, column_names.index('w')]
                if rank == 0:
                    logging.info(f"Using {column_names[column_names.index('w')]} as the WEIGHT column")
            else:
                data_cat['WEIGHT'] = 1.0
                if rank == 0:
                    logging.info("WEIGHT column does not exist in the list. Setting WEIGHT to 1.0")

            # Free memory
            del data_arr
            gc.collect()

        elif data_ext == "fits":
            # Read .fits file and process it
            data_cat = FITSCatalog(data_path)
            xyz_columns = [col for col in ['x', 'y', 'z', 'X', 'Y', 'Z'] if col in data_cat.columns]

            # Check for position columns
            if len(xyz_columns) == 3:
                data_cat['Position'] = np.column_stack([np.array(data_cat[col]) for col in xyz_columns])
                if rank == 0:
                    logging.info(f"Using {xyz_columns} as the position columns")

            # Check for weight columns
            weight_columns = [col for col in ['w', 'W', 'WEIGHT'] if col in data_cat.columns]
            if weight_columns:
                data_cat['WEIGHT'] = data_cat[weight_columns[0]]
                if rank == 0:
                    logging.info(f"Using {weight_columns[0]} as the WEIGHT column")
            else:
                data_cat['WEIGHT'] = 1.0
                if rank == 0:
                    logging.info("WEIGHT column does not exist in the list. Setting WEIGHT to 1.0")

        return data_cat

    elif geometry == "survey-like":
        # Handle survey-like geometry
        data_path = catalogs.get("data_path")
        randoms_path = catalogs.get("randoms_path")
        data_ext = data_path.split('.')[-1].lower() if data_path else None
        randoms_ext = randoms_path.split('.')[-1].lower() if randoms_path else None

        # Check if file types are supported and match
        if data_ext not in supported_types or randoms_ext not in supported_types:
            raise ValueError(f"Unsupported file type: {data_ext} or {randoms_ext}")
        if data_ext != randoms_ext:
            raise ValueError("Data and randoms file types must match for survey-like geometry")

        if data_ext == "npy":
            data_arr = npy_reader(data_path, comm)
            randoms_arr = npy_reader(randoms_path, comm)

            # Create ArrayCatalogs for data and randoms
            data_cat = ArrayCatalog({"Position": data_arr[:, :3]})
            randoms_cat = ArrayCatalog({"Position": randoms_arr[:, :3]})
            if rank == 0:
                logging.info(f"Using {column_names[:3]} as the position columns for both data and randoms")

            # Check for weight and nz columns
            if 'w_comp' in column_names and 'w_fkp' in column_names and 'nz' in column_names:
                data_cat['WEIGHT'] = data_arr[:, column_names.index('w_comp')]
                randoms_cat['WEIGHT'] = randoms_arr[:, column_names.index('w_comp')]
                data_cat['WEIGHT_FKP'] = data_arr[:, column_names.index('w_fkp')]
                randoms_cat['WEIGHT_FKP'] = randoms_arr[:, column_names.index('w_fkp')]
                data_cat['NZ'] = data_arr[:, column_names.index('nz')]
                randoms_cat['NZ'] = randoms_arr[:, column_names.index('nz')]
                if rank == 0:
                    logging.info(f"Using {column_names[column_names.index('w_comp')]} as the WEIGHT column, \
                                 {column_names[column_names.index('w_fkp')]} as the WEIGHT_FKP column, \
                                 and {column_names[column_names.index('nz')]} as the NZ column for both data and randoms")
            else:
                if rank == 0:
                    raise ValueError("WEIGHT, WEIGHT_FKP, or NZ column does not exist in the list or wrong column name in the yaml file")
                    
            # Free memory
            del data_arr, randoms_arr
            gc.collect()

        elif data_ext == "fits":
            data_cat = FITSCatalog(data_path)
            randoms_cat = FITSCatalog(randoms_path)

            # Filter data and randoms by redshift range
            ZMIN, ZMAX = z_range
            data_cat = data_cat[(data_cat['Z'] > ZMIN) & (data_cat['Z'] < ZMAX)]
            randoms_cat = randoms_cat[(randoms_cat['Z'] > ZMIN) & (randoms_cat['Z'] < ZMAX)]

            # Define cosmology
            cosmo = cosmology.Cosmology(
                h=para_cosmo["h"],
                Omega0_b=para_cosmo["Omega0_b"],
                Omega0_cdm=para_cosmo["Omega0_cdm"]
            ).match(para_cosmo["sigma8"])
            # cosmo = cosmology.Cosmology(h=0.676).match(Omega0_m=0.31)

            if rank == 0:
                logging.info(f"Using cosmology: {cosmo}")

            # Transform sky coordinates to Cartesian coordinates
            data_cat['Position'] = transform.SkyToCartesian(data_cat['RA'], data_cat['DEC'], data_cat['Z'], cosmo=cosmo)
            randoms_cat['Position'] = transform.SkyToCartesian(randoms_cat['RA'], randoms_cat['DEC'], randoms_cat['Z'], cosmo=cosmo)

            # Add completeness weights according to the specified scheme
            data_cat['WEIGHT'] = add_completeness_weight(data_cat, comp_weight_plan, "data", comm)
            randoms_cat['WEIGHT'] = add_completeness_weight(randoms_cat, comp_weight_plan, "randoms", comm)


        return data_cat, randoms_cat

    else:
        # Raise error for unsupported geometry types
        raise ValueError(f"Unsupported geometry type: {geometry}. Only 'box-like' and 'survey-like' are supported.")



def npy_reader(data_path,comm):
    """
    Read npy files in parallel using MPI by automatically handle the file splitting.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        arr = format.open_memmap(data_path, mode='r')
        shape = arr.shape
        dtype = arr.dtype
    else:
        shape = None
        dtype = None

    # Broadcast shape and dtype to all ranks
    shape = comm.bcast(shape, root=0)
    dtype = comm.bcast(dtype, root=0)

    # Calculate the number of rows for each rank
    total_rows = shape[0]
    rows_per_rank = total_rows // size
    remainder = total_rows % size
    start_row = rank * rows_per_rank + min(rank, remainder)
    end_row = start_row + rows_per_rank + (1 if rank < remainder else 0)

    # Each rank reads its own slice of the data
    local_data = format.open_memmap(data_path, mode='r')[start_row:end_row]

    # logging the shape of the data read by each rank
    logging.info(f"Reading .npy data from disk, Rank {rank}: Rows {start_row} to {end_row-1}, shape {local_data.shape}")

    return local_data





def add_completeness_weight(dr, comp_weight_plan, catalog_type, comm):
    """
    Adds a completeness weight column to the given DataFrame based on the specified scheme.

    Parameters:
        dr (DataFrame): The input catalog (data or randoms) to which the completeness weight will be added.
        comp_weight_plan (dict): A dictionary containing the following keys:
            - "name_alias" (str): The column name to use as the completeness weight if it exists in `dr`.
            - "scheme" (str): The weighting scheme to use if the alias is not found. 
                Supported values are "boss" and "eboss".
        catalog_type (str): The type of catalog ("data" or "randoms").
        comm (MPI.Comm): The MPI communicator.

    Returns:
        Series: The completeness weight column added to the catalog.

    Raises:
        ValueError: If an invalid scheme is provided in `comp_weight_plan["scheme"]`.

    Logic:
        1. Retrieve the rank of the current process using the MPI communicator.
        2. Log the start of the completeness weight addition process for the given catalog type.
        3. Check if the alias specified in `comp_weight_plan["name_alias"]` exists in the catalog columns:
            - If it exists, use it as the completeness weight and log the action.
        4. If the alias does not exist, check if a column named 'WEIGHT' already exists:
            - If it exists, use it as the completeness weight and log the action.
        5. If neither the alias nor 'WEIGHT' exists, check for the presence of the columns 
           'WEIGHT_SYSTOT', 'WEIGHT_NOZ', and 'WEIGHT_CP':
            - If the scheme is "boss", compute the completeness weight as:
              WEIGHT = WEIGHT_SYSTOT * (WEIGHT_NOZ + WEIGHT_CP - 1.0)
              Log the use of the BOSS-like scheme.
            - If the scheme is "eboss", compute the completeness weight as:
              WEIGHT = WEIGHT_SYSTOT * WEIGHT_NOZ * WEIGHT_CP
              Log the use of the eBOSS-like scheme.
            - If the scheme is invalid, log an error and raise a ValueError.
        6. If none of the above conditions are met, set the completeness weight to 1.0 and log the action.
        7. Return the computed completeness weight column.
    """
    rank = comm.Get_rank()
    if rank == 0:
        logging.info(f"Adding completeness weight for {catalog_type} catalog in rank {rank}")

    alias = comp_weight_plan.get("name_alias") 
    scheme = comp_weight_plan.get("scheme")

    if alias in dr.columns:
        dr['WEIGHT'] = dr[alias]
        if rank == 0:
            logging.info(f"Using {alias} as the completeness weight")
    elif 'WEIGHT' in dr.columns:
        if rank == 0:
            logging.info("Using existing WEIGHT column")
    elif all(col in dr.columns for col in ['WEIGHT_SYSTOT', 'WEIGHT_NOZ', 'WEIGHT_CP']):
        if scheme == "boss":
            dr['WEIGHT'] = dr['WEIGHT_SYSTOT'] * (dr['WEIGHT_NOZ'] + dr['WEIGHT_CP'] - 1.0)
            if rank == 0:
                logging.info("Using BOSS-like completeness weight")
        elif scheme == "eboss":
            dr['WEIGHT'] = dr['WEIGHT_SYSTOT'] * dr['WEIGHT_NOZ'] * dr['WEIGHT_CP']
            if rank == 0:
                logging.info("Using eBOSS-like completeness weight")
        else:
            if rank == 0:
                logging.error("Invalid scheme for completeness weight")
            raise ValueError("Invalid scheme for completeness weight")
    else:
        dr['WEIGHT'] = 1.0
        if rank == 0:
            logging.info("Setting completeness weight to 1.0")

    return dr['WEIGHT']
