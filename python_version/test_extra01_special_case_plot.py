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
from lib.special_cases import FFPEHalfAlpha


DEFAULT_CONFIG_PATH = os.path.join( REPO_FOLDER, 'configs', 'test_extra01_special_case_plot.json' )
DEFAULT_OUTPUT_PATH = os.path.join( CURRENT_FOLDER, 'test_extra01_special_case_plot.pdf' )


def ensure_directory( file_path ):
    os.makedirs( os.path.dirname( file_path ), exist_ok = True )


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

    half_alpha = FFPEHalfAlpha(
        D_o = float( config[ 'D_o' ] ),
        D_f = float( config[ 'D_f' ] )
    )

    timer = time.perf_counter()
    for i, t in enumerate( t_values ):
        solution_values[ i, : ] = half_alpha.get_value( Y[ i, : ], float( t ), int( config[ 'd' ] ) )
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
