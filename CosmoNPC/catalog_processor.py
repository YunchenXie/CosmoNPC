import numpy as np
from numpy.lib import format
from mpi4py import MPI
import logging
# from nbodykit.lab import *
import warnings
import gc
import fitsio


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



def fits_reader(comm, files, column_names):
    """
    Read FITS files in parallel across all MPI ranks.
    
    Args:
        comm: MPI communicator
        files: List of FITS file paths
        column_names: List of column names to read (raises error if any are missing)
        
    Returns:
        numpy.ndarray: Combined data from all files for this rank's row ranges
    """
    rank, size = comm.Get_rank(), comm.Get_size()

    # Initialize empty array with correct dtype
    dtype = [(col, 'f8') for col in column_names]
    result = np.array([], dtype=dtype)
    
    for f in files:
        with fitsio.FITS(f) as fits:
            # Select appropriate HDU (prefer binary table)
            hdu = fits[1] if len(fits) > 1 else fits[0]
            nrows = hdu.get_nrows()

            if rank == 0:
                logging.info(f"File: {f}, Total Rows: {nrows}, All Columns: {hdu.get_colnames()}")

            # Calculate row distribution
            base, extra = nrows // size, nrows % size
            start = rank * base + min(rank, extra)
            end = start + base + (1 if rank < extra else 0)
            
            # Read data if this rank has rows
            if start < nrows:
                data = hdu.read(rows=range(start, end), columns=column_names)
                
                # Verify all columns are present
                if not all(col in data.dtype.names for col in column_names):
                    missing = [col for col in column_names if col not in data.dtype.names]
                    raise ValueError(f"Missing columns in {f}: {missing}")
                
                # Concatenate directly
                result = np.concatenate([result, data]) if len(result) > 0 else data

    # Log the number of rows read by this rank
    local_row_count = len(result)
    total_row_count = comm.reduce(local_row_count, op=MPI.SUM, root=0)
    
    if rank == 0:
        logging.info(f"Total rows across all ranks: {total_row_count}")

    return result



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


def catalog_reader(catalog, geometry, column_names, z_range, comp_weight_plan, para_cosmo, comm):
    """
    Reads the data/randoms catalog and applies necessary preprocessing steps.
    Returns:
        DataFrame: The processed catalog.
    """
    # Initialize MPI
    rank = comm.Get_rank()
    size = comm.Get_size()

    supported_types = {"npy", "fits"}  # Supported file types for catalogs
    data_path = catalog  # could be a single file or a list of files

    # Determine the file extension
    if isinstance(data_path, list):
        data_ext = data_path[0].split('.')[-1].lower() if data_path and data_path[0] else None
    else:
        data_ext = data_path.split('.')[-1].lower() if data_path else None

    # Check if the file type is supported
    if data_ext not in supported_types:
        raise ValueError(f"Unsupported data file type: {data_ext}")

    if geometry == "box-like":
        """
            Handle box-like geometry, currently only single .npy file is supported
        """
        if data_ext == "npy":

            assert column_names is not None and len(column_names) >= 3, \
                "For box-like geometry with .npy file, column_names must be provided \
                    with at least 3 elements for x,y,z."

            # Read .npy file and process it
            data_arr = npy_reader(data_path, comm)
            position_indices = [column_names.index(axis) for axis in ['x', 'y', 'z']]
            if rank == 0:
                logging.info(f"Using {column_names} as the position columns")
            # Create a structured ndarray directly without intermediate arrays
            data_cat = np.zeros(len(data_arr), dtype=[('Position', 'f8', (3,)), ('WEIGHT', 'f8')])
            data_cat['Position'] = np.column_stack([data_arr[:, idx] for idx in position_indices])

            # Check for weight column and assign directly
            if 'w' in column_names:
                data_cat['WEIGHT'] = data_arr[:, column_names.index('w')]
                if rank == 0:
                    logging.info(f"Using {column_names[column_names.index('w')]} as the WEIGHT column")
            else:
                data_cat['WEIGHT'] = 1.0
                if rank == 0:
                    logging.info("WEIGHT column does not exist in the list. Setting WEIGHT to 1.0")


        elif data_ext == "fits":
            data_arr = fits_reader(comm, catalog, column_names)

            # Create a structured ndarray directly using the keys from data_arr
            data_cat = np.zeros(len(data_arr), dtype=[('Position', 'f8', (3,)), ('WEIGHT', 'f8')])
            data_cat['Position'] = np.column_stack([data_arr['x'], data_arr['y'], data_arr['z']])

            # Assign the weight column
            if 'w' in data_arr.dtype.names:
                data_cat['WEIGHT'] = data_arr['w']
                if rank == 0:
                    logging.info("Using 'w' as the WEIGHT column")
            else:
                data_cat['WEIGHT'] = 1.0
                if rank == 0:
                    logging.info("WEIGHT column does not exist in the data. Setting WEIGHT to 1.0")

        # Free memory
        del data_arr
        gc.collect()

        return data_cat
    
    elif geometry == "survey-like":
        """
        Handle survey-like geometry, note that the catalog could be either data or randoms
        """

        if data_ext == "npy":

            assert column_names is not None and len(column_names) >= 5, \
                "For survey-like geometry with .npy file, column_names must be provided \
                    with at least 5 elements for x, y, z, w_fkp, w_comp."
            
            # Read .npy file and process it
            data_arr = npy_reader(data_path, comm)

            #Create ArrayCatalogs
            position_indices = [column_names.index(axis) for axis in ['x', 'y', 'z']]
            data_cat = ArrayCatalog({"Position": np.column_stack([data_arr[:, idx] for idx in position_indices])})

            # Check for weight and nz columns
            if 'w_comp' in column_names and 'w_fkp' in column_names :
                data_cat['WEIGHT'] = data_arr[:, column_names.index('w_comp')]
                data_cat['WEIGHT_FKP'] = data_arr[:, column_names.index('w_fkp')]
                if rank == 0:
                    logging.info(f"Using {column_names[column_names.index('w_comp')]} as the WEIGHT column, \
                                 {column_names[column_names.index('w_fkp')]} as the WEIGHT_FKP column.")
            if 'nz' in column_names:
                data_cat['NZ'] = data_arr[:, column_names.index('nz')]
                if rank == 0:
                    logging.info(f"Using {column_names[column_names.index('nz')]} as the NZ column.")
            else:
                data_cat['NZ'] = 1.0
                if rank == 0:
                    logging.info("NZ column does not exist in the list. Setting NZ to 1.0")

            # Free memory
            del data_arr
            gc.collect()

        elif data_ext == "fits":
            full_data_cat = FITSCatalog(data_path) 

            # Create a new catalog with only the necessary columns
            data_cat = ArrayCatalog({"RA": full_data_cat["RA"],
                                    "DEC": full_data_cat["DEC"],
                                    "Z": full_data_cat["Z"],
                                    "WEIGHT": full_data_cat["WEIGHT"],
                                    "WEIGHT_FKP": full_data_cat["WEIGHT_FKP"]})
            if "NZ" in full_data_cat.columns:
                data_cat["NZ"] = full_data_cat["NZ"]
            elif "NX" in full_data_cat.columns:
                # for DESI-like catalog
                data_cat["NZ"] = full_data_cat["NX"]
            else:
                data_cat["NZ"] = 1.0
                if rank == 0:
                    logging.info("NZ column does not exist in the catalog. Setting NZ to 1.0")

            # free memory
            del full_data_cat
            gc.collect()

            # cut the catalog based on z_range
            ZMIN, ZMAX = z_range
            data_cat = data_cat[(data_cat['Z'] > ZMIN) & (data_cat['Z'] < ZMAX)]

            # Fiducial cosmology setup
            cosmo = cosmology.Cosmology(
                h=para_cosmo["h"],
                Omega0_b=para_cosmo["Omega0_b"],
                Omega0_cdm=para_cosmo["Omega0_cdm"]
            ).match(para_cosmo["sigma8"])
            #  cosmo = cosmology.Cosmology(h=0.676).match(Omega0_m=0.31)

            if rank == 0:
                logging.info(f"Using fiducial cosmology: {cosmo}")

            # Convert (RA, DEC, Z) to (x, y, z) 
            data_cat['Position'] = \
                transform.SkyToCartesian(data_cat['RA'], data_cat['DEC'], data_cat['Z'], cosmo=cosmo)
            
            # Add completeness weights according to the specified scheme
            data_cat['WEIGHT'] = add_completeness_weight(data_cat, comp_weight_plan, "data", comm)

        return data_cat          


            