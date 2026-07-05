import numpy as np
from functools import lru_cache
from numpy.polynomial.legendre import legval
from scipy.special import eval_legendre


@lru_cache( maxsize = 128 )
def get_legendre_basis_coefficient( n ):
    coefficient = np.zeros( int( n ) + 1, dtype = np.float64 )
    coefficient[ int( n ) ] = 1.0
    return coefficient


def evaluate_legendre_polynomial( n, x ):
    n = int( n )
    x_array = np.asarray( x )
    if x_array.ndim > 0 and x_array.size > 32 and n >= 32:
        return legval( x_array, get_legendre_basis_coefficient( n ) )
    return eval_legendre( n, x )


class LegendrePolynomial:
    def get_polynomial( self, n ):
        n = int( n )

        def polynomial( x ):
            return evaluate_legendre_polynomial( n, x )

        return polynomial

    def get_polynomial_coefficient( self, n ):
        polynomial = np.polynomial.legendre.Legendre.basis( int( n ) )
        power_polynomial = polynomial.convert( kind = np.polynomial.Polynomial )
        return power_polynomial.coef.copy()

    def get_polynomial_coefficient_special( self, n ):
        coefficient = self.get_polynomial_coefficient( n )
        nonzero_index = np.flatnonzero( np.abs( coefficient ) > 0 )
        return coefficient[ nonzero_index ]
