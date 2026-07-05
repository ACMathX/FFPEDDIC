import math
from functools import lru_cache

import numpy as np
import torch
from numpy.polynomial.legendre import legvander
from scipy.integrate import quad
from scipy.special import eval_legendre
from scipy.special import roots_jacobi
from scipy.special import roots_legendre


@lru_cache( maxsize = 64 )
def get_unit_legendre_points( n ):
    n = int( n )
    if n >= 100:
        return get_unit_legendre_points_by_chebfun_asymptotics( n )
    return roots_legendre( n )


def get_unit_legendre_points_by_chebfun_asymptotics( n ):
    m = int( math.ceil( n / 2.0 ) )
    j0_roots = get_besselj_zero_roots( m )

    vn = 1.0 / ( n + 0.5 )
    a = j0_roots * vn
    u = 1.0 / np.tan( a )
    ua = u * a
    u2 = u ** 2
    a2 = a ** 2

    F0 = a
    F1 = 0.125 * ( u * a - 1.0 ) / a
    if n < 10000:
        a3 = a ** 3
        F2 = (
            ( 6.0 * a2 * ( 1.0 + u2 ) + 25.0 - u * ( 31.0 * u2 + 33.0 ) * a3 )
            / ( 384.0 * a3 )
        )
    else:
        F2 = 0.0
    if n < 1000:
        u4 = u ** 4
        a5 = a ** 5
        R30 = u * ( 2595.0 + 6350.0 * u2 + 3779.0 * u4 ) / 15360.0
        R31 = - ( 31.0 * u2 + 11.0 ) / 1024.0
        R32 = u / 512.0
        R33 = -25.0 / 3072.0
        R35 = -1073.0 / 5120.0
        F3 = R30 + R35 / a5 + ( 1.0 + u2 ) * ( R31 / a + R32 / a2 + R33 / a3 )
    else:
        F3 = 0.0

    theta = F0 + F1 * vn ** 2 + F2 * vn ** 4 + F3 * vn ** 6
    positive_x = np.cos( theta )

    W0 = 1.0
    W1 = 0.125 * ( ua + a2 - 1.0 ) / a2
    if n < 10000:
        a3 = a ** 3
        a4 = a2 ** 2
        u4 = u ** 4
        W2 = (
            81.0
            - 31.0 * ua
            - 3.0 * ( 1.0 - 2.0 * u2 ) * a2
            + 6.0 * u * a3
            - ( 27.0 + 84.0 * u2 + 56.0 * u4 ) * a4
        ) / ( 384.0 * a4 )
    else:
        W2 = 0.0
    if n < 1000:
        u3 = u ** 3
        u4 = u ** 4
        u5 = u ** 5
        u6 = u3 ** 2
        a3 = a ** 3
        a4 = a2 ** 2
        a5 = a ** 5
        a6 = a3 ** 2
        Q30 = 187.0 / 96.0 * u4 + 295.0 / 256.0 * u2 + 151.0 / 160.0 * u6 + 153.0 / 1024.0
        Q31 = -119.0 / 768.0 * u3 - 35.0 / 384.0 * u5 - 65.0 / 1024.0 * u
        Q32 = 5.0 / 512.0 + 7.0 / 384.0 * u4 + 15.0 / 512.0 * u2
        Q33 = u3 / 512.0 - 13.0 / 1536.0 * u
        Q34 = -7.0 / 384.0 * u2 + 53.0 / 3072.0
        Q35 = 3749.0 / 15360.0 * u
        Q36 = -1125.0 / 1024.0
        W3 = Q30 + Q31 / a + Q32 / a2 + Q33 / a3 + Q34 / a4 + Q35 / a5 + Q36 / a6
    else:
        W3 = 0.0

    Jk2 = evaluate_besselj1_squared_at_j0_roots( m )
    positive_w = 2.0 / (
        ( Jk2 / vn ** 2 )
        * ( a / np.sin( a ) )
        * ( W0 + W1 * vn ** 2 + W2 * vn ** 4 + W3 * vn ** 6 )
    )

    if n % 2 == 0:
        x = np.concatenate( ( - positive_x, positive_x[ :: -1 ] ) )
        w = np.concatenate( ( positive_w, positive_w[ :: -1 ] ) )
    else:
        x = np.concatenate( ( - positive_x[ : -1 ], np.array( [ 0.0 ] ), positive_x[ -2 :: -1 ] ) )
        w = np.concatenate( ( positive_w, positive_w[ -2 :: -1 ] ) )

    return x.astype( np.float64 ), w.astype( np.float64 )


def get_besselj_zero_roots( n ):
    root_index = np.arange( 1, int( n ) + 1, dtype = np.float64 )
    beta = 0.25 * ( 4.0 * root_index - 1.0 ) * math.pi

    coefficients = np.array(
        [
            576.0 * 423748443625564327.0 / 13059009124761600.0,
            0.0,
            720.0 * ( -8249725736393.0 ) / 10463949619200.0,
            0.0,
            144.0 * 2092163573.0 / 11890851840.0,
            0.0,
            6.0 * ( -6277237.0 ) / 20643840.0,
            0.0,
            4.0 * 3779.0 / 61440.0,
            0.0,
            -31.0 / 384.0,
            0.0,
            1.0 / 8.0,
            0.0,
        ],
        dtype = np.float64
    )
    roots = beta + np.polyval( coefficients, 1.0 / beta )

    first_roots = np.array(
        [
            2.4048255576957728,
            5.5200781102863106,
            8.6537279129110122,
            11.791534439014281,
            14.930917708487785,
            18.071063967910922,
            21.211636629879258,
            24.352471530749302,
            27.493479132040254,
            30.634606468431975,
            33.775820213573568,
            36.917098353664044,
            40.058425764628239,
            43.199791713176730,
            46.341188371661814,
            49.482609897397817,
            52.624051841114996,
            55.765510755019979,
            58.906983926080942,
            62.048469190227170,
        ],
        dtype = np.float64
    )
    roots[ : min( int( n ), first_roots.size ) ] = first_roots[ : min( int( n ), first_roots.size ) ]
    return roots


def evaluate_besselj1_squared_at_j0_roots( n ):
    values = np.zeros( int( n ), dtype = np.float64 )
    first_values = np.array(
        [
            0.2695141239419169,
            0.1157801385822037,
            0.07368635113640822,
            0.05403757319811628,
            0.04266142901724309,
            0.03524210349099610,
            0.03002107010305467,
            0.02614739149530809,
            0.02315912182469139,
            0.02078382912226786,
        ],
        dtype = np.float64
    )
    values[ : min( int( n ), first_values.size ) ] = first_values[ : min( int( n ), first_values.size ) ]
    if n <= first_values.size:
        return values

    root_index = np.arange( first_values.size + 1, int( n ) + 1, dtype = np.float64 )
    ak = math.pi * ( root_index - 0.25 )
    ak2inv = 1.0 / ak ** 2
    c1 = -171497088497.0 / 15206400.0
    c2 = 461797.0 / 1152.0
    c3 = -172913.0 / 8064.0
    c4 = 151.0 / 80.0
    c5 = -7.0 / 24.0
    values[ first_values.size: ] = (
        1.0
        / ( math.pi * ak )
        * ( 2.0 + ak2inv ** 2 * ( c5 + ak2inv * ( c4 + ak2inv * ( c3 + ak2inv * ( c2 + ak2inv * c1 ) ) ) ) )
    )
    return values


@lru_cache( maxsize = 128 )
def get_unit_jacobi_points( n, beta ):
    return roots_jacobi( int( n ), 0.0, float( beta ) )


def clear_quadrature_caches():
    get_unit_legendre_points.cache_clear()
    get_unit_jacobi_points.cache_clear()


def get_legendre_points( n, interval ):
    a, b = interval
    x, w = get_unit_legendre_points( int( n ) )
    s = 0.5 * ( b - a ) * x + 0.5 * ( a + b )
    w = 0.5 * ( b - a ) * w
    return s.astype( np.float64 ), w.astype( np.float64 )


def get_jacobi_points( n, beta, interval ):
    a, b = interval
    x, w = get_unit_jacobi_points( int( n ), float( beta ) )
    s = 0.5 * ( b - a ) * ( x + 1.0 ) + a
    w = ( 0.5 * ( b - a ) ) ** ( float( beta ) + 1.0 ) * w
    return s.astype( np.float64 ), w.astype( np.float64 )


def as_tensor_pair( s, w, dtype, device ):
    s_tensor = torch.as_tensor( s, dtype = dtype, device = device )
    w_tensor = torch.as_tensor( w, dtype = dtype, device = device )
    return s_tensor, w_tensor


class FractionalQuadrature:
    def __init__( self, alpha, delta_t, L = 1.0 ):
        self.alpha = float( alpha )
        self.delta_t = float( delta_t )
        self.L = float( L )
        self.interval = ( 0.0, self.L )
        self.validate_parameters()

    def validate_parameters( self ):
        if self.alpha < 0:
            raise ValueError( 'alpha must be nonnegative.' )
        if self.delta_t < 0:
            raise ValueError( 'delta_t must be nonnegative.' )
        if self.L <= 0:
            raise ValueError( 'L must be positive.' )

    def get_weights_by_exactness( self, n, eps = 1e-14 ):
        beta = 2.0 * self.alpha
        s, _ = get_legendre_points( n, self.interval )
        y = 2.0 * s / self.L - 1.0

        F = np.zeros( int( n ), dtype = np.float64 )

        for k in range( int( n ) ):
            def integrand( x, polynomial_index = k ):
                return (
                    math.exp( - abs( x ) ** beta * self.delta_t )
                    * eval_legendre( polynomial_index, 2.0 * x / self.L - 1.0 )
                )

            F[ k ], _ = quad(
                integrand,
                0.0,
                self.L,
                epsabs = eps,
                epsrel = 100.0 * eps,
                limit = 1000
            )
        A = legvander( y, int( n ) - 1 ).T

        w = np.linalg.solve( A, F )
        return s, w

    def compute_by_exactness( self, n, f, dtype, device ):
        s, w = self.get_weights_by_exactness( n )
        s_tensor, w_tensor = as_tensor_pair( s, w, dtype, device )
        return torch.sum( w_tensor * f( s_tensor ) )

    def compute_by_expansion( self, n, f, number_of_term, dtype, device ):
        s, w = get_legendre_points( n, self.interval )
        s_tensor, w_tensor = as_tensor_pair( s, w, dtype, device )
        value = torch.sum( w_tensor * f( s_tensor ) )

        beta = 2.0 * self.alpha
        multiplier = 1.0
        for k in range( 1, int( number_of_term ) + 1 ):
            multiplier = - multiplier * self.delta_t / k
            current_beta = k * beta
            if current_beta <= 5.0:
                s_jacobi, w_jacobi = get_jacobi_points( n, current_beta, self.interval )
                s_current, w_current = as_tensor_pair( s_jacobi, w_jacobi, dtype, device )
                value = value + multiplier * torch.sum( w_current * f( s_current ) )
            else:
                value = value + multiplier * torch.sum(
                    w_tensor * ( torch.abs( s_tensor ) ** current_beta ) * f( s_tensor )
                )
        return value

    def get_value( self, n, f, dtype, device ):
        if self.delta_t <= 1e-3:
            return self.compute_by_expansion( n, f, 4, dtype, device )
        if self.delta_t <= 1e-2:
            return self.compute_by_expansion( n, f, 6, dtype, device )
        if self.delta_t <= 1e-1:
            return self.compute_by_expansion( n, f, 9, dtype, device )
        if self.delta_t <= 1e0:
            return self.compute_by_expansion( n, f, 17, dtype, device )
        return self.compute_by_exactness( n, f, dtype, device )
