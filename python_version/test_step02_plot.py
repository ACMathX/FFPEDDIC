import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


CURRENT_FOLDER = os.path.dirname( os.path.abspath( __file__ ) )
REPO_FOLDER = os.path.dirname( CURRENT_FOLDER )
if CURRENT_FOLDER not in sys.path:
    sys.path.insert( 0, CURRENT_FOLDER )

from lib.config import load_config
from lib.defaults import get_local_solver_parameters
from lib.solver import FFPESolver


DEFAULT_CONFIG_PATH = os.path.join( REPO_FOLDER, 'configs', 'test_step02_plot.json' )
DEFAULT_OUTPUT_PATH = os.path.join( CURRENT_FOLDER, 'test_step02_plot.pdf' )


def ensure_directory( file_path ):
    os.makedirs( os.path.dirname( file_path ), exist_ok = True )


def build_solver_parameters( config ):
    return get_local_solver_parameters( config )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--config', type = str, default = DEFAULT_CONFIG_PATH )
    parser.add_argument( '--output', type = str, default = DEFAULT_OUTPUT_PATH )
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config( args.config )

    y_values = np.linspace( 0.0, float( config[ 'y_limit' ] ), int( config[ 'y_number' ] ) )
    t_values = np.linspace( 0.0, float( config[ 't_limit' ] ), int( config[ 't_number' ] ) )[ 1 : ]
    Y, T = np.meshgrid( y_values, t_values )
    solution_values = np.zeros_like( Y )

    timer = time.perf_counter()
    for i, t in enumerate( t_values ):
        solver = FFPESolver(
            d = int( config[ 'd' ] ),
            alpha = float( config[ 'alpha' ] ),
            D_o = float( config[ 'D_o' ] ),
            D_f = float( config[ 'D_f' ] ),
            t = float( t ),
            parameters = build_solver_parameters( config )
        )
        solver.general_initialization()
        for j, y in enumerate( y_values ):
            solution_values[ i, j ] = solver.get_value( float( y ) ).value
    total_seconds = time.perf_counter() - timer

    ensure_directory( args.output )
    fig = plt.figure( figsize = ( 10, 7 ) )
    axis = fig.add_subplot( 111, projection = '3d' )
    axis.plot_surface( Y, T, solution_values, cmap = 'viridis' )
    axis.set_xlabel( 'y' )
    axis.set_ylabel( 't' )
    axis.set_zlabel( 'solution' )
    fig.tight_layout()
    fig.savefig( args.output, dpi = 300 )
    plt.close( fig )

    print( 'total_seconds = {:.6f}'.format( total_seconds ) )
    print( 'output = {}'.format( args.output ) )


if __name__ == '__main__':
    main()
