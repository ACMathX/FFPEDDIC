import math

import mpmath as mp
import numpy as np
from scipy.special import erfcx as scipy_erfcx
from scipy.special import gamma


def evaluate_scaled_complementary_error_function( z ):
    return scipy_erfcx( z )


def get_n_sphere_surface_area( n ):
    half_n_plus_1 = ( mp.mpf( n ) + 1 ) / 2
    return 2 * mp.pi ** half_n_plus_1 / mp.gamma( half_n_plus_1 )


def as_array( y ):
    return np.atleast_1d( np.asarray( y, dtype = np.float64 ) )


def scalar_or_array( original, value ):
    if np.isscalar( original ):
        return float( np.asarray( value ).reshape( -1 )[ 0 ] )
    return value


def hyper( a, b, z ):
    return mp.hyper( [ mp.mpf( item ) for item in a ], [ mp.mpf( item ) for item in b ], mp.mpf( z ) )


def hyper_borel( a, b, z ):
    return mp.hyper(
        [ mp.mpf( item ) for item in a ],
        [ mp.mpf( item ) for item in b ],
        mp.mpf( z ),
        maxterms = 10 ** 6,
        maxprec = 10000,
        asymp_tol = mp.mpf( '1e-20' )
    )


class FFPEHalfAlpha:
    def __init__( self, D_o = 1.0, D_f = 1.0 ):
        self.D_o = float( D_o )
        self.D_f = float( D_f )

    def get_value( self, y, t, d ):
        if self.D_o == 0:
            return self.get_value_zero_D_o( y, t, d )

        y_array = as_array( y )
        value = np.zeros_like( y_array, dtype = np.complex128 )

        zero_position = y_array == 0
        positive_position = y_array > 0
        if np.any( zero_position ):
            value[ zero_position ] = self.get_value_positive_D_o_zero_y( t, d )
        if np.any( positive_position ):
            value[ positive_position ] = self.get_value_positive_D_o( y_array[ positive_position ], t, d )
        value = np.real( value )
        return scalar_or_array( y, value )

    def get_value_positive_D_o( self, y, t, d ):
        if d % 2 == 0:
            raise ValueError( 'Positive D_o half-alpha formula requires odd d.' )

        y_array = as_array( y )
        component_base = 2.0 * math.sqrt( self.D_o * t )
        P_component = ( self.D_f * t - y_array * 1j ) / component_base
        M_component = ( self.D_f * t + y_array * 1j ) / component_base

        if d == 1:
            value = (
                evaluate_scaled_complementary_error_function( P_component )
                + evaluate_scaled_complementary_error_function( M_component )
            ) / (
                2.0 * math.sqrt( 4.0 * math.pi * self.D_o * t )
            )
            return np.real( value )

        q_limit = ( d + 1 ) // 2
        K_recursion_base = 2.0 * self.D_o * t
        KP_recursion = - ( self.D_f * t - y_array * 1j ) / K_recursion_base
        KM_recursion = - ( self.D_f * t + y_array * 1j ) / K_recursion_base

        KP = np.zeros( y_array.shape + ( q_limit, ), dtype = np.complex128 )
        KM = np.zeros( y_array.shape + ( q_limit, ), dtype = np.complex128 )

        KP[ ..., 0 ] = math.sqrt( math.pi ) / component_base * evaluate_scaled_complementary_error_function( P_component )
        KP[ ..., 1 ] = 1.0 / K_recursion_base + KP_recursion * KP[ ..., 0 ]
        KM[ ..., 0 ] = math.sqrt( math.pi ) / component_base * evaluate_scaled_complementary_error_function( M_component )
        KM[ ..., 1 ] = 1.0 / K_recursion_base + KM_recursion * KM[ ..., 0 ]

        for q in range( 2, q_limit ):
            KP[ ..., q ] = KP_recursion * KP[ ..., q - 1 ] + ( q - 1 ) / K_recursion_base * KP[ ..., q - 2 ]
            KM[ ..., q ] = KM_recursion * KM[ ..., q - 1 ] + ( q - 1 ) / K_recursion_base * KM[ ..., q - 2 ]

        memoize_cache = {}

        def T( p, q ):
            key = ( int( p ), int( q ) )
            if key in memoize_cache:
                return memoize_cache[ key ]
            q_index = int( q ) - 1
            if p == 1:
                value = np.real( ( KP[ ..., q_index ] - KM[ ..., q_index ] ) / ( y_array * 1j ) )
            elif p == 3:
                value = - ( KP[ ..., q_index - 1 ] + KM[ ..., q_index - 1 ] )
                value = value + ( KP[ ..., q_index - 2 ] - KM[ ..., q_index - 2 ] ) / ( y_array * 1j )
                value = np.real( value * 2.0 / ( y_array ** 2 ) )
            else:
                value = - ( p - 3 ) * T( p - 4, q - 2 ) + ( p - 2 ) * T( p - 2, q - 2 )
                value = np.real( value / y_array * ( p - 1 ) / y_array )
            memoize_cache[ key ] = value
            return value

        value = T( d - 2, d - 1 )
        value = value / ( ( 2.0 * math.pi ) ** d )
        value = value * 2.0 * ( math.pi ** ( ( d - 1 ) / 2.0 ) ) / math.gamma( ( d - 1 ) / 2.0 )
        return np.real( value )

    def get_value_positive_D_o_zero_y( self, t, d ):
        component_base = 2.0 * math.sqrt( self.D_o * t )
        component = self.D_f * t / component_base

        if d == 1:
            return evaluate_scaled_complementary_error_function( component ) / math.sqrt( 4.0 * math.pi * self.D_o * t )

        K_recursion_base = 2.0 * self.D_o * t
        K_recursion = - self.D_f / ( 2.0 * self.D_o )
        K = np.zeros( int( d ), dtype = np.float64 )
        K[ 0 ] = (
            math.sqrt( math.pi )
            / ( 2.0 * math.sqrt( self.D_o * t ) )
            * evaluate_scaled_complementary_error_function( component )
        )
        K[ 1 ] = 1.0 / K_recursion_base + K_recursion * K[ 0 ]

        for q in range( 2, int( d ) ):
            K[ q ] = K_recursion * K[ q - 1 ] + ( q - 1 ) / K_recursion_base * K[ q - 2 ]

        value = K[ int( d ) - 1 ] / ( ( 2.0 * math.pi ) ** d )
        value = value * 2.0 * ( math.pi ** ( d / 2.0 ) ) / math.gamma( d / 2.0 )
        return value

    def get_value_zero_D_o( self, y, t, d ):
        y_array = as_array( y )
        tau = self.D_f * t
        k = ( d + 1.0 ) / 2.0
        value = gamma( k ) / ( math.pi ** k ) * tau / ( ( tau ** 2 + y_array ** 2 ) ** k )
        return scalar_or_array( y, value )


class FFPEOneThirdAlpha:
    def __init__( self, D_o = 0.0, D_f = 1.0 ):
        self.D_o = float( D_o )
        self.D_f = float( D_f )

    def get_value( self, y, t, d ):
        if self.D_o != 0:
            raise NotImplementedError( 'Positive D_o is not implemented for alpha = 1 / 3 special cases.' )
        return self.get_value_zero_D_o( y, t, d )

    def get_value_zero_D_o( self, y, t, d ):
        y_array = as_array( y )
        tau = mp.mpf( self.D_f * t )
        d_high = mp.mpf( d )
        d_h = d_high / 2
        values = []
        for y_value in y_array.reshape( -1 ):
            z = - ( mp.mpf( y_value ) ** 2 ) / ( tau ** 3 ) * ( 3 ** 3 ) / ( 2 ** 2 )
            coefficient = mp.gamma( d_h * 3 ) / mp.gamma( d_h )
            coefficient = coefficient * 3 / ( 2 ** d_high ) / ( mp.pi ** d_h ) / ( tau ** ( d_h * 3 ) )
            values.append( float( hyper( [ d_h + mp.mpf( 1 ) / 3, d_h + mp.mpf( 2 ) / 3 ], [], z ) * coefficient ) )
        value = np.asarray( values, dtype = np.float64 ).reshape( y_array.shape )
        return scalar_or_array( y, value )


class FFPETwoThirdsAlpha:
    def __init__( self, D_o = 0.0, D_f = 1.0 ):
        self.D_o = float( D_o )
        self.D_f = float( D_f )

    def get_value( self, y, t, d ):
        if self.D_o != 0:
            raise NotImplementedError( 'Positive D_o is not implemented for alpha = 2 / 3 special cases.' )
        return self.get_value_zero_D_o( y, t, d )

    def get_value_zero_D_o( self, y, t, d ):
        y_array = as_array( y )
        tau = mp.mpf( self.D_f * t )
        d_high = mp.mpf( d )
        d_4 = d_high / 4
        d_4_2 = ( d_high + 2 ) / 4
        values = []
        for y_value in y_array.reshape( -1 ):
            y_high = mp.mpf( y_value )
            z = ( y_high ** 4 ) / ( tau ** 3 ) * ( 3 ** 3 ) / ( 4 ** 4 )
            value_1 = hyper( [ d_4 + mp.mpf( 1 ) / 3, d_4 + mp.mpf( 2 ) / 3 ], [ mp.mpf( 1 ) / 2, d_4_2 ], z )
            value_1 = value_1 * mp.gamma( d_4 * 3 ) / mp.gamma( d_4 * 2 )
            value_2 = hyper( [ d_4_2 + mp.mpf( 1 ) / 3, d_4_2 + mp.mpf( 2 ) / 3 ], [ mp.mpf( 3 ) / 2, d_4 + 1 ], z )
            value_2 = value_2 * mp.gamma( d_4_2 * 3 ) / mp.gamma( d_4_2 * 2 )
            value_2 = value_2 * ( y_high / 2 ) ** 2 / ( tau ** ( mp.mpf( 3 ) / 2 ) )
            coefficient = 3 / ( 2 ** ( d_high + 1 ) ) / ( mp.pi ** ( d_high / 2 ) ) / ( tau ** ( d_4 * 3 ) )
            values.append( float( ( value_1 - value_2 ) * coefficient ) )
        value = np.asarray( values, dtype = np.float64 ).reshape( y_array.shape )
        return scalar_or_array( y, value )


class FFPERationalAlpha:
    def __init__( self, p = 1, q = 2, D_o = 0.0, D_f = 1.0, digits = 50, fallback_parameters = None ):
        if math.gcd( int( p ), int( q ) ) != 1:
            raise ValueError( 'p and q must be coprime.' )
        if p >= q:
            raise ValueError( 'p / q must be less than 1.' )
        self.p = int( p )
        self.q = int( q )
        self.D_o = float( D_o )
        self.D_f = float( D_f )
        self.digits = int( digits )
        self.fallback_parameters = fallback_parameters

    def get_value( self, y, t, d ):
        if self.D_o != 0:
            raise NotImplementedError( 'Positive D_o is not implemented for rational alpha special cases.' )
        return self.get_value_zero_D_o( y, t, d )

    def get_value_zero_D_o( self, y, t, d ):
        if self.p == 1 and self.q == 2:
            return FFPEHalfAlpha( self.D_o, self.D_f ).get_value_zero_D_o( y, t, d )
        if self.p == 1 and self.q == 3:
            return FFPEOneThirdAlpha( self.D_o, self.D_f ).get_value_zero_D_o( y, t, d )
        if self.p == 2 and self.q == 3:
            return FFPETwoThirdsAlpha( self.D_o, self.D_f ).get_value_zero_D_o( y, t, d )
        if self.q > 8:
            return self.get_value_zero_D_o_by_solver_fallback( y, t, d )

        mp.mp.dps = self.digits
        y_array = as_array( y )
        tau = mp.mpf( self.D_f * t )
        d_high = mp.mpf( d )
        d_h_high = d_high / 2
        p_high = mp.mpf( self.p )
        q_high = mp.mpf( self.q )
        r_high = p_high / q_high
        values = []

        for y_value in y_array.reshape( -1 ):
            y_high = mp.mpf( y_value )
            z = - ( y_high / 2 ) ** 2 / ( tau ** ( 1 / r_high ) )
            coefficient = 1 / ( tau ** ( d_h_high / r_high ) * ( 2 * mp.pi ) ** d_high )
            c = ( z / ( p_high ** 2 ) ) ** p_high * q_high ** q_high
            value = mp.mpf( 0 )
            factorial_value = mp.mpf( 1 )

            for i in range( self.p ):
                if i > 1:
                    factorial_value = factorial_value * i
                i_high = mp.mpf( i )
                f = get_n_sphere_surface_area( d + 2 * i - 1 )
                f = f * mp.gamma( ( d_h_high + i_high ) / r_high + 1 )
                f = f * ( z / mp.pi ) ** i_high
                f = f / ( factorial_value * ( d_high + 2 * i ) )

                a = [ mp.mpf( item ) / q_high + ( d_h_high + i_high ) / p_high for item in range( 1, self.q ) ]
                b1 = [ ( mp.mpf( item ) + i_high + 1 ) / p_high for item in range( 0, self.p - i - 1 ) ]
                b1 += [ ( mp.mpf( item ) + i_high + 1 ) / p_high for item in range( self.p - i, self.p ) ]
                b2 = [ ( mp.mpf( item ) + d_h_high + i_high ) / p_high for item in range( 1, self.p ) ]
                value = value + hyper_borel( a, b1 + b2, c ) * f

            values.append( float( value * coefficient ) )

        value = np.asarray( values, dtype = np.float64 ).reshape( y_array.shape )
        return scalar_or_array( y, value )

    def get_value_zero_D_o_by_solver_fallback( self, y, t, d ):
        try:
            from .solver import FFPESolver
        except ImportError:
            from lib.solver import FFPESolver

        try:
            from .defaults import get_local_solver_parameters
        except ImportError:
            from lib.defaults import get_local_solver_parameters

        parameters = get_local_solver_parameters()
        if self.fallback_parameters is not None:
            parameters.update( self.fallback_parameters )

        solver = FFPESolver(
            d = int( d ),
            alpha = self.p / self.q,
            D_o = 0.0,
            D_f = self.D_f,
            t = float( t ),
            parameters = parameters
        )
        solver.general_initialization()

        y_array = as_array( y )
        values = []
        for y_value in y_array.reshape( -1 ):
            values.append( solver.get_value( float( y_value ) ).value )
        value = np.asarray( values, dtype = np.float64 ).reshape( y_array.shape )
        return scalar_or_array( y, value )
