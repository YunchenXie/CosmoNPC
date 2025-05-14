
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
    ZMIN, ZMAX = z_range

    # Load data and random catalogs
    data = FITSCatalog(catalogs["data_path"])
    randoms = FITSCatalog(catalogs["randoms_path"])

    # Filter randoms within the redshift range
    randoms = randoms[(randoms['Z'] > ZMIN) & (randoms['Z'] < ZMAX)]

    # Filter data within the redshift range
    data = data[(data['Z'] > ZMIN) & (data['Z'] < ZMAX)]

    # Specify the cosmology
    # cosmo = cosmology.Cosmology(
    #     h=para_cosmo["h"],
    #     Omega0_b=para_cosmo["Omega0_b"],
    #     Omega0_cdm=para_cosmo["Omega0_cdm"]
    # ).match(para_cosmo["sigma8"])

    cosmo = cosmology.Cosmology(h=0.676).match(Omega0_m=0.31)


    # Add Cartesian position column to data and randoms
    data['Position'] = transform.SkyToCartesian(
        data['RA'], data['DEC'], data['Z'], cosmo=cosmo
    )
    randoms['Position'] = transform.SkyToCartesian(
        randoms['RA'], randoms['DEC'], randoms['Z'], cosmo=cosmo
    )

    # Add the Completeness Weights
    """
    The completeness weights may depend on particular catalogs.
    """
    # Add completeness weights using the provided function
    data['WEIGHT'] = add_completeness_weight(data, comp_weight_plan, "data", comm)
    randoms['WEIGHT'] = add_completeness_weight(randoms, comp_weight_plan, "randoms", comm)

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
