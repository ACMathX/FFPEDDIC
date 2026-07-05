LOCAL_SOLVER_PARAMETERS = {
    'L' : 1.0,
    'M_ini' : 80.0,
    'gamma' : 0.5,
    'd_tol' : 1e-14,
    'M_lim' : 5121.0,
    'window_multiplier' : 50.0,
    'window_points_cap' : None,
    'device' : 'cpu',
}


def get_local_solver_parameters( config = None ):
    parameters = LOCAL_SOLVER_PARAMETERS.copy()
    if config is None:
        return parameters

    for key in parameters.keys():
        if key in config:
            parameters[ key ] = config[ key ]

    parameters[ 'L' ] = float( parameters[ 'L' ] )
    parameters[ 'M_ini' ] = float( parameters[ 'M_ini' ] )
    parameters[ 'gamma' ] = float( parameters[ 'gamma' ] )
    parameters[ 'd_tol' ] = float( parameters[ 'd_tol' ] )
    parameters[ 'M_lim' ] = float( parameters[ 'M_lim' ] )
    parameters[ 'window_multiplier' ] = float( parameters[ 'window_multiplier' ] )
    if parameters[ 'window_points_cap' ] is not None:
        parameters[ 'window_points_cap' ] = int( parameters[ 'window_points_cap' ] )
    return parameters
