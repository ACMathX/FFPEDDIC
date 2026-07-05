import argparse
import os
import sys
import time

import numpy as np


CURRENT_FOLDER = os.path.dirname( os.path.abspath( __file__ ) )
REPO_FOLDER = os.path.dirname( CURRENT_FOLDER )
if CURRENT_FOLDER not in sys.path:
    sys.path.insert( 0, CURRENT_FOLDER )

from lib.config import load_config
from lib.defaults import get_local_solver_parameters
from lib.solver import FFPESolver
from lib.special_cases import FFPERationalAlpha


DEFAULT_CONFIG_PATH = os.path.join( REPO_FOLDER, 'configs', 'test_step01_accuracy_verification.json' )


def build_solver_parameters( config ):
    return get_local_solver_parameters( config )


def evaluate_one_case( config ):
    alpha = float( config[ 'alpha_numerator' ] ) / float( config[ 'alpha_denominator' ] )
    solver = FFPESolver(
        d = int( config[ 'd' ] ),
        alpha = alpha,
        D_o = float( config[ 'D_o' ] ),
        D_f = float( config[ 'D_f' ] ),
        t = float( config[ 'delta_t' ] ),
        parameters = build_solver_parameters( config )
    )

    initialization_timer = time.perf_counter()
    solver.general_initialization()
    initialization_seconds = time.perf_counter() - initialization_timer

    evaluation_timer = time.perf_counter()
    approximate_result = solver.get_value( float( config[ 'y' ] ) )
    evaluation_seconds = time.perf_counter() - evaluation_timer

    reference = FFPERationalAlpha(
        p = int( config[ 'alpha_numerator' ] ),
        q = int( config[ 'alpha_denominator' ] ),
        D_o = float( config[ 'D_o' ] ),
        D_f = float( config[ 'D_f' ] ),
        digits = int( config.get( 'digits', 50 ) )
    )
    reference_timer = time.perf_counter()
    reference_value = reference.get_value(
        y = float( config[ 'y' ] ),
        t = float( config[ 'delta_t' ] ),
        d = int( config[ 'd' ] )
    )
    reference_seconds = time.perf_counter() - reference_timer

    relative_error = abs( approximate_result.value - reference_value ) / max( abs( reference_value ), sys.float_info.min )
    return {
        'approximate_value' : approximate_result.value,
        'reference_value' : reference_value,
        'relative_error' : relative_error,
        'convergence_flag' : approximate_result.convergence_flag,
        'value_2_difference' : approximate_result.value_2_difference,
        'initialization_seconds' : initialization_seconds,
        'evaluation_seconds' : evaluation_seconds,
        'reference_seconds' : reference_seconds,
    }


def evaluate_grid_case( config ):
    alpha = float( config[ 'alpha_numerator' ] ) / float( config[ 'alpha_denominator' ] )
    y_values = np.linspace( 0.0, float( config[ 'y_limit' ] ), int( config[ 'y_number' ] ) )
    t_values = np.linspace( 0.0, float( config[ 't_limit' ] ), int( config[ 't_number' ] ) )[ 1 : ]

    reference = FFPERationalAlpha(
        p = int( config[ 'alpha_numerator' ] ),
        q = int( config[ 'alpha_denominator' ] ),
        D_o = float( config[ 'D_o' ] ),
        D_f = float( config[ 'D_f' ] ),
        digits = int( config.get( 'digits', 50 ) )
    )

    max_relative_error = 0.0
    rows = []
    total_timer = time.perf_counter()
    for t in t_values:
        solver = FFPESolver(
            d = int( config[ 'd' ] ),
            alpha = alpha,
            D_o = float( config[ 'D_o' ] ),
            D_f = float( config[ 'D_f' ] ),
            t = float( t ),
            parameters = build_solver_parameters( config )
        )
        solver.general_initialization()
        for y in y_values:
            result = solver.get_value( float( y ) )
            reference_value = reference.get_value( float( y ), float( t ), int( config[ 'd' ] ) )
            relative_error = abs( result.value - reference_value ) / max( abs( reference_value ), sys.float_info.min )
            max_relative_error = max( max_relative_error, relative_error )
            rows.append( ( float( t ), float( y ), result.value, reference_value, relative_error ) )
    total_seconds = time.perf_counter() - total_timer
    return rows, max_relative_error, total_seconds


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--config', type = str, default = DEFAULT_CONFIG_PATH )
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config( args.config )
    test_type = int( config.get( 'test_type', 1 ) )

    if test_type == 1:
        result = evaluate_one_case( config )
        print( 'approximate_value = {:.16e}'.format( result[ 'approximate_value' ] ) )
        print( 'reference_value = {:.16e}'.format( result[ 'reference_value' ] ) )
        print( 'relative_error = {:.9e}'.format( result[ 'relative_error' ] ) )
        print( 'convergence_flag = {}'.format( result[ 'convergence_flag' ] ) )
        print( 'initialization_seconds = {:.6f}'.format( result[ 'initialization_seconds' ] ) )
        print( 'evaluation_seconds = {:.6f}'.format( result[ 'evaluation_seconds' ] ) )
        print( 'reference_seconds = {:.6f}'.format( result[ 'reference_seconds' ] ) )
    else:
        rows, max_relative_error, total_seconds = evaluate_grid_case( config )
        print( 'rows = {}'.format( len( rows ) ) )
        print( 'max_relative_error = {:.9e}'.format( max_relative_error ) )
        print( 'total_seconds = {:.6f}'.format( total_seconds ) )


if __name__ == '__main__':
    main()
