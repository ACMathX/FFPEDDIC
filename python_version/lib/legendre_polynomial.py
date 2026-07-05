import numpy as np
from scipy.special import eval_legendre


RECURRENCE_MINIMUM_ARRAY_SIZE = 8192
RECURRENCE_MINIMUM_ORDER = 16
HORNER_MINIMUM_ARRAY_SIZE = 8192
HORNER_MAXIMUM_ORDER = 20
MATLAB_IMPLEMENTED_LIMIT = 36


def validate_legendre_order( n ):
    n = int( n )
    if n < 0:
        raise ValueError( 'n must be nonnegative.' )
    return n


def make_readonly_copy( values ):
    values = np.array( values, dtype = np.float64, copy = True )
    values.setflags( write = False )
    return values


def multiply_power_coefficient_by_x( coefficient ):
    result = np.zeros( coefficient.size + 1, dtype = np.float64 )
    result[ 1: ] = coefficient
    return result


def multiply_power_coefficient_by_two_x_minus_one( coefficient ):
    result = np.zeros( coefficient.size + 1, dtype = np.float64 )
    result[ 1: ] = 2.0 * coefficient
    result[ : -1 ] -= coefficient
    return result


def evaluate_power_polynomial_by_horner( coefficient, x ):
    x_array = np.asarray( x )
    result = np.zeros_like( x_array, dtype = np.result_type( x_array.dtype, np.float64 ) )
    result += coefficient[ -1 ]
    for coefficient_index in range( coefficient.size - 2, -1, -1 ):
        result *= x_array
        result += coefficient[ coefficient_index ]
    return result


def evaluate_legendre_polynomial_by_recurrence( n, x ):
    n = validate_legendre_order( n )
    x_array = np.asarray( x )
    result_dtype = np.result_type( x_array.dtype, np.float64 )

    if n == 0:
        return np.ones_like( x_array, dtype = result_dtype )
    if n == 1:
        return np.array( x_array, dtype = result_dtype, copy = True )

    P_n_minus_2 = np.ones_like( x_array, dtype = result_dtype )
    P_n_minus_1 = np.array( x_array, dtype = result_dtype, copy = True )
    P_n = np.empty_like( P_n_minus_1 )

    for polynomial_index in range( 2, n + 1 ):
        np.multiply( x_array, P_n_minus_1, out = P_n )
        P_n *= 2 * polynomial_index - 1
        P_n -= ( polynomial_index - 1 ) * P_n_minus_2
        P_n /= polynomial_index
        P_n_minus_2, P_n_minus_1, P_n = P_n_minus_1, P_n, P_n_minus_2

    return P_n_minus_1


def evaluate_legendre_polynomial( n, x ):
    n = validate_legendre_order( n )
    x_array = np.asarray( x )
    if (
        x_array.ndim > 0
        and x_array.size >= RECURRENCE_MINIMUM_ARRAY_SIZE
        and n >= RECURRENCE_MINIMUM_ORDER
    ):
        return evaluate_legendre_polynomial_by_recurrence( n, x_array )
    return eval_legendre( n, x )


class LegendrePolynomial:
    def __init__( self, implemented_limit = MATLAB_IMPLEMENTED_LIMIT ):
        self.implemented_limit = int( implemented_limit )
        self.polynomial_cache = {
            0 : self.order_00_polynomial(),
            1 : self.order_01_polynomial(),
        }
        self.polynomial_coefficient_cache = {
            0 : make_readonly_copy( [ 1.0 ] ),
            1 : make_readonly_copy( [ 0.0, 1.0 ] ),
        }
        self.polynomial_coefficient_special_cache = {
            0 : make_readonly_copy( [ 1.0 ] ),
            1 : make_readonly_copy( [ -1.0, 2.0 ] ),
        }

    def get_polynomial( self, n ):
        n = validate_legendre_order( n )
        if n not in self.polynomial_cache:
            if n < self.implemented_limit:
                coefficient = self.get_polynomial_coefficient( n )

                def polynomial( x ):
                    x_array = np.asarray( x )
                    if (
                        n <= HORNER_MAXIMUM_ORDER
                        and x_array.ndim > 0
                        and x_array.size >= HORNER_MINIMUM_ARRAY_SIZE
                    ):
                        return evaluate_power_polynomial_by_horner( coefficient, x_array )
                    return evaluate_legendre_polynomial( n, x )
            else:
                def polynomial( x ):
                    return evaluate_legendre_polynomial( n, x )

            self.polynomial_cache[ n ] = polynomial

        return self.polynomial_cache[ n ]

    def get_polynomial_coefficient( self, n ):
        n = validate_legendre_order( n )
        if n not in self.polynomial_coefficient_cache:
            factor_1 = 2 * n - 1
            factor_2 = n - 1
            C_n_minus_1 = self.get_polynomial_coefficient( n - 1 )
            C_n_minus_2 = self.get_polynomial_coefficient( n - 2 )

            C = (
                factor_1 * multiply_power_coefficient_by_x( C_n_minus_1 )
                - factor_2 * np.pad( C_n_minus_2, ( 0, 2 ) )
            ) / n
            self.polynomial_coefficient_cache[ n ] = make_readonly_copy( C )

        return self.polynomial_coefficient_cache[ n ].copy()

    def get_polynomial_coefficient_special( self, n ):
        n = validate_legendre_order( n )
        if n not in self.polynomial_coefficient_special_cache:
            factor_1 = 2 * n - 1
            factor_2 = n - 1
            C_n_minus_1 = self.get_polynomial_coefficient_special( n - 1 )
            C_n_minus_2 = self.get_polynomial_coefficient_special( n - 2 )

            C = (
                factor_1 * multiply_power_coefficient_by_two_x_minus_one( C_n_minus_1 )
                - factor_2 * np.pad( C_n_minus_2, ( 0, 2 ) )
            ) / n
            self.polynomial_coefficient_special_cache[ n ] = make_readonly_copy( C )

        return self.polynomial_coefficient_special_cache[ n ].copy()

    @staticmethod
    def order_00_polynomial():
        def polynomial( x ):
            return np.ones_like( np.asarray( x ), dtype = np.result_type( np.asarray( x ).dtype, np.float64 ) )

        return polynomial

    @staticmethod
    def order_01_polynomial():
        def polynomial( x ):
            return np.asarray( x )

        return polynomial
