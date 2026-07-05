import math

import numpy as np
import torch
from scipy.special import jv

from .quadrature import FractionalQuadrature
from .quadrature import as_tensor_pair
from .quadrature import get_legendre_points
from .windowing import TestWindowingFunction02


_WINDOWING_CACHE = {}


def clear_solver_caches():
    _WINDOWING_CACHE.clear()


class FFPESolverResult:
    def __init__( self, value, convergence_flag, value_2_difference ):
        self.value = float( value )
        self.convergence_flag = bool( convergence_flag )
        self.value_2_difference = float( value_2_difference )


def compute_sphere_surface_area( n ):
    return 2.0 * math.pi ** ( ( float( n ) + 1.0 ) / 2.0 ) / math.gamma( ( float( n ) + 1.0 ) / 2.0 )


def evaluate_integer_besselj( order, x ):
    if order < 0:
        raise ValueError( 'Integer Bessel order must be nonnegative.' )
    if order == 0:
        return torch.special.bessel_j0( x )
    if order == 1:
        return torch.special.bessel_j1( x )

    small_flag = torch.abs( x ) < 1e-12
    safe_x = torch.where( small_flag, torch.ones_like( x ), x )

    J_previous = torch.special.bessel_j0( safe_x )
    J_current = torch.special.bessel_j1( safe_x )
    for current_order in range( 1, int( order ) ):
        J_next = 2.0 * current_order / safe_x * J_current - J_previous
        J_previous = J_current
        J_current = J_next

    zero_value = torch.zeros_like( x )
    return torch.where( small_flag, zero_value, J_current )


def evaluate_half_integer_besselj( order_index, x ):
    small_flag = torch.abs( x ) < 1e-12
    safe_x = torch.where( small_flag, torch.ones_like( x ), x )
    factor = torch.sqrt(
        torch.as_tensor( 2.0 / math.pi, dtype = x.dtype, device = x.device ) / safe_x
    )

    J_previous = factor * torch.cos( safe_x ) # J_{ -1 / 2 }
    J_current = factor * torch.sin( safe_x ) # J_{ 1 / 2 }

    if order_index == 0:
        return torch.where( small_flag, torch.zeros_like( x ), J_current )

    nu = 0.5
    for _ in range( int( order_index ) ):
        J_next = 2.0 * nu / safe_x * J_current - J_previous
        J_previous = J_current
        J_current = J_next
        nu = nu + 1.0

    return torch.where( small_flag, torch.zeros_like( x ), J_current )


def evaluate_besselj_for_dimension( d, x ):
    if d == 1:
        raise ValueError( 'd = 1 uses cosine pathway, not Bessel pathway.' )
    order = ( d - 2 ) / 2.0
    values = jv( order, x.detach().cpu().numpy() )
    values = np.asarray( values, dtype = np.float64 )
    return torch.as_tensor( values, dtype = x.dtype, device = x.device )


class FFPESolver:
    def __init__(
        self,
        d,
        alpha,
        D_o,
        D_f,
        t,
        parameters = None
    ):
        if parameters is None:
            parameters = {}

        self.d = int( d )
        self.alpha = float( alpha )
        self.D_o = float( D_o )
        self.D_f = float( D_f )
        self.t = float( t )

        self.dtype = parameters.get( 'dtype', torch.float64 )
        device_name = parameters.get( 'device', 'cpu' )
        self.device = torch.device( device_name )

        self.L = float( parameters.get( 'L', 1.0 ) )
        self.M_ini = float( parameters.get( 'M_ini', 80.0 ) )
        self.gamma = float( parameters.get( 'gamma', 0.5 ) )
        self.d_tol = float( parameters.get( 'd_tol', 1e-14 ) )
        self.M_lim = float( parameters.get( 'M_lim', 5121.0 ) )
        self.window_multiplier = float( parameters.get( 'window_multiplier', 50.0 ) )
        self.window_points_cap = parameters.get( 'window_points_cap', None )

        self.y1 = math.pi / 2.0
        self.y2 = math.pi / 2.0
        self.t1 = 0.1

        self.d_r = 0.0 if self.d == 1 else ( self.d - 2 ) / 2.0
        self.d_h = self.d / 2.0
        self.d_m = self.d - 1
        self.coef = self.compute_coefficient( self.d_m )

        self.s1 = None
        self.w1 = None
        self.s2 = []
        self.w2 = []
        self.v2 = []
        self.g2 = []

        self.validate_parameters()

    def validate_parameters( self ):
        if self.d <= 0:
            raise ValueError( 'd must be positive.' )
        if self.alpha <= 0:
            raise ValueError( 'alpha must be positive.' )
        if self.D_o < 0 or self.D_f < 0:
            raise ValueError( 'Diffusion coefficients must be nonnegative.' )
        if self.t <= 0:
            raise ValueError( 't must be positive.' )
        if self.L <= 0:
            raise ValueError( 'L must be positive.' )
        if self.M_ini >= self.M_lim:
            raise ValueError( 'M_ini must be smaller than M_lim.' )
        if self.gamma <= 0 or self.gamma >= 1:
            raise ValueError( 'gamma must satisfy 0 < gamma < 1.' )
        if self.d_tol <= 0:
            raise ValueError( 'd_tol must be positive.' )

    @classmethod
    def compute_coefficient( cls, n ):
        return compute_sphere_surface_area( n ) / ( 2.0 * math.pi )

    @classmethod
    def from_config( cls, config ):
        alpha = config.get( 'alpha', None )
        if alpha is None:
            alpha = float( config[ 'alpha_numerator' ] ) / float( config[ 'alpha_denominator' ] )

        parameters = config.get( 'solver_parameters', {} ).copy()
        for key in [ 'L', 'M_ini', 'gamma', 'd_tol', 'M_lim', 'window_multiplier', 'window_points_cap' ]:
            if key in config:
                parameters[ key ] = config[ key ]
        if 'device' in config:
            parameters[ 'device' ] = config[ 'device' ]

        return cls(
            d = config[ 'd' ],
            alpha = alpha,
            D_o = config[ 'D_o' ],
            D_f = config[ 'D_f' ],
            t = config[ 't' ],
            parameters = parameters
        )

    def get_window_point_count( self, M ):
        N = int( round( self.window_multiplier * M ) )
        if self.window_points_cap is not None:
            N = min( N, int( self.window_points_cap ) )
        return max( N, 1 )

    def windowing_function_initialization( self ):
        cache_key = (
            self.L,
            self.M_ini,
            self.M_lim,
            self.gamma,
            self.window_multiplier,
            self.window_points_cap,
            str( self.dtype ),
            str( self.device ),
        )
        if cache_key in _WINDOWING_CACHE:
            cached_s2, cached_w2, cached_v2 = _WINDOWING_CACHE[ cache_key ]
            self.s2 = list( cached_s2 )
            self.w2 = list( cached_w2 )
            self.v2 = list( cached_v2 )
            return

        self.s2 = []
        self.w2 = []
        self.v2 = []

        M = self.M_ini
        while M < self.M_lim:
            windowing_function = TestWindowingFunction02( M, self.gamma )
            N = self.get_window_point_count( M )
            s, w = get_legendre_points( N, ( self.L, M ) )
            s_tensor, w_tensor = as_tensor_pair( s, w, self.dtype, self.device )

            self.s2.append( s_tensor )
            self.w2.append( w_tensor )
            self.v2.append( windowing_function.get_value( s_tensor ) )
            M = M * 2.0

        _WINDOWING_CACHE[ cache_key ] = ( tuple( self.s2 ), tuple( self.w2 ), tuple( self.v2 ) )

    def quadrature_initialization( self, n = 16, L = 1.0, eps = 1e-14 ):
        self.L = float( L )
        FQ = FractionalQuadrature( self.alpha, self.D_f * self.t, self.L )
        s, w = FQ.get_weights_by_exactness( n, eps )
        self.s1, self.w1 = as_tensor_pair( s, w, self.dtype, self.device )

    def update_g2( self ):
        self.ensure_windowing_initialized()
        if self.d == 1:
            g = self.get_g_1d( self.D_o, self.D_f, self.alpha, self.t )
        else:
            g = self.get_g( self.D_o, self.D_f, self.alpha, self.t )
        self.g2 = []
        for s, v in zip( self.s2, self.v2 ):
            self.g2.append( g( s ) * v )

    def general_initialization( self ):
        self.windowing_function_initialization()
        self.quadrature_initialization( 16, self.L )
        self.update_g2()

    def ensure_windowing_initialized( self ):
        if len( self.s2 ) == 0 or len( self.w2 ) == 0 or len( self.v2 ) == 0:
            raise ValueError( 'Call windowing_function_initialization or general_initialization before evaluation.' )

    def ensure_quadrature_initialized( self ):
        if self.s1 is None or self.w1 is None:
            raise ValueError( 'Call quadrature_initialization or general_initialization before evaluation.' )

    def ensure_general_initialized( self ):
        self.ensure_windowing_initialized()
        self.ensure_quadrature_initialized()
        if len( self.g2 ) == 0:
            raise ValueError( 'Call update_g2 or general_initialization before evaluation.' )

    def get_value_1d( self, y ):
        self.ensure_general_initialized()
        f = self.get_f_1d( y, self.D_o, self.t )
        g_complement = self.get_g_complement_1d( y )

        convergence_flag = False
        last_value_2 = float( 'inf' )
        current_value_2 = torch.tensor( 0.0, dtype = self.dtype, device = self.device )
        value_2_difference = float( 'inf' )

        for s2_, w2_, g2_ in zip( self.s2, self.w2, self.g2 ):
            current_value_2 = torch.sum( w2_ * g_complement( s2_ ) * g2_ )
            value_2_difference = abs( float( current_value_2 ) - last_value_2 )
            if value_2_difference < self.d_tol:
                convergence_flag = True
                break
            last_value_2 = float( current_value_2 )

        value_1 = torch.sum( self.w1 * f( self.s1 ) )
        value = ( current_value_2 + value_1 ) / math.pi
        return FFPESolverResult( value, convergence_flag, value_2_difference )

    def get_value_hd( self, y ):
        self.ensure_general_initialized()
        f = self.get_f( y, self.D_o, self.t )
        g_complement = self.get_g_complement( y )

        convergence_flag = False
        last_value_2 = float( 'inf' )
        current_value_2 = torch.tensor( 0.0, dtype = self.dtype, device = self.device )
        value_2_difference = float( 'inf' )

        for s2_, w2_, g2_ in zip( self.s2, self.w2, self.g2 ):
            current_value_2 = torch.sum( w2_ * g_complement( s2_ ) * g2_ )
            value_2_difference = abs( float( current_value_2 ) - last_value_2 )
            if value_2_difference < self.d_tol:
                convergence_flag = True
                break
            last_value_2 = float( current_value_2 )

        value_1 = torch.sum( self.w1 * f( self.s1 ) )
        value = ( current_value_2 + value_1 ) / ( float( y ) ** self.d_r )
        return FFPESolverResult( value, convergence_flag, value_2_difference )

    def get_value_plain_1d( self, y, t, D_o ):
        self.ensure_windowing_initialized()
        f = self.get_f_1d( y, D_o, t )
        p_hat = self.get_p_hat_1d( y, D_o, self.D_f, self.alpha, t )

        convergence_flag = False
        last_value_2 = float( 'inf' )
        current_value_2 = torch.tensor( 0.0, dtype = self.dtype, device = self.device )
        value_2_difference = float( 'inf' )

        for s2_, w2_, v2_ in zip( self.s2, self.w2, self.v2 ):
            current_value_2 = torch.sum( w2_ * p_hat( s2_ ) * v2_ )
            value_2_difference = abs( float( current_value_2 ) - last_value_2 )
            if value_2_difference < self.d_tol:
                convergence_flag = True
                break
            last_value_2 = float( current_value_2 )

        FQ = FractionalQuadrature( self.alpha, self.D_f * t, self.L )
        value_1 = FQ.get_value( 16, f, self.dtype, self.device )
        value = ( current_value_2 + value_1 ) / math.pi
        return FFPESolverResult( value, convergence_flag, value_2_difference )

    def get_value_plain_hd( self, y, t, D_o ):
        self.ensure_windowing_initialized()
        f = self.get_f( y, D_o, t )
        p_hat = self.get_p_hat( y, D_o, self.D_f, self.alpha, t )

        convergence_flag = False
        last_value_2 = float( 'inf' )
        current_value_2 = torch.tensor( 0.0, dtype = self.dtype, device = self.device )
        value_2_difference = float( 'inf' )

        for s2_, w2_, v2_ in zip( self.s2, self.w2, self.v2 ):
            current_value_2 = torch.sum( w2_ * p_hat( s2_ ) * v2_ )
            value_2_difference = abs( float( current_value_2 ) - last_value_2 )
            if value_2_difference < self.d_tol:
                convergence_flag = True
                break
            last_value_2 = float( current_value_2 )

        FQ = FractionalQuadrature( self.alpha, self.D_f * t, self.L )
        value_1 = FQ.get_value( 16, f, self.dtype, self.device )
        value = ( current_value_2 + value_1 ) / ( float( y ) ** self.d_r )
        return FFPESolverResult( value, convergence_flag, value_2_difference )

    def get_value_zero_displacement_plain( self, t, D_o ):
        self.ensure_windowing_initialized()
        f = self.get_f_zero_displacement( D_o, t )
        p_hat = self.get_p_hat_zero_displacement( D_o, self.D_f, self.alpha, t )

        convergence_flag = False
        last_value_2 = float( 'inf' )
        current_value_2 = torch.tensor( 0.0, dtype = self.dtype, device = self.device )
        value_2_difference = float( 'inf' )

        for s2_, w2_, v2_ in zip( self.s2, self.w2, self.v2 ):
            current_value_2 = torch.sum( w2_ * p_hat( s2_ ) * v2_ )
            value_2_difference = abs( float( current_value_2 ) - last_value_2 )
            if value_2_difference < self.d_tol:
                convergence_flag = True
                break
            last_value_2 = float( current_value_2 )

        FQ = FractionalQuadrature( self.alpha, self.D_f * t, self.L )
        value_1 = FQ.get_value( 16, f, self.dtype, self.device )
        value = ( current_value_2 + value_1 ) * self.coef
        return FFPESolverResult( value, convergence_flag, value_2_difference )

    def get_value_zero_displacement_with_scaling( self ):
        if abs( self.D_o ) < 1e-12:
            value = ( self.D_f * self.t ) ** ( - self.d / ( 2.0 * self.alpha ) )
            value = value * math.gamma( self.d / ( 2.0 * self.alpha ) + 1.0 )
            value = value / ( ( 2.0 * math.pi ) ** self.d )
            value = value / self.d * 2.0 * math.pi ** ( self.d / 2.0 ) / math.gamma( self.d / 2.0 )
            return FFPESolverResult( value, True, 0.0 )

        result = self.get_value_zero_displacement_plain( self.t, self.D_o )
        if result.convergence_flag:
            return result

        target_D_o = 10.0
        T = ( self.D_o / target_D_o ) ** ( 1.0 / ( 1.0 - 1.0 / self.alpha ) ) * self.t
        scale = ( self.t / T ) ** ( - 1.0 / ( 2.0 * self.alpha ) )
        result = self.get_value_zero_displacement_plain( T, target_D_o )
        return FFPESolverResult(
            result.value * scale ** self.d,
            result.convergence_flag,
            result.value_2_difference
        )

    def get_value_no_fractional_diffusion( self, y ):
        if self.D_o < 1e-14:
            raise ValueError( 'At least one diffusion coefficient must be positive.' )
        denominator_1 = 4.0 * self.D_o * self.t
        denominator_2 = ( denominator_1 * math.pi ) ** self.d_h
        value = math.exp( - float( y ) ** 2 / denominator_1 ) / denominator_2
        return FFPESolverResult( value, True, 0.0 )

    def get_value_with_scaling_1d( self, y ):
        if abs( y ) < 1e-14:
            return self.get_value_zero_displacement_with_scaling()

        result = self.get_value_1d( y )
        if result.convergence_flag:
            return result

        target_y = 0.0
        if y < self.y1:
            target_y = self.y1
        elif y > self.y2:
            target_y = self.y2

        if target_y > 0:
            T = ( target_y / y ) ** ( 2.0 * self.alpha ) * self.t
            scale = target_y / y
            result = self.get_value_plain_1d(
                target_y,
                T,
                ( self.t / T ) ** ( 1.0 - 1.0 / self.alpha ) * self.D_o
            )
            return FFPESolverResult(
                result.value * scale,
                result.convergence_flag,
                result.value_2_difference
            )

        return result

    def get_value_with_scaling_hd( self, y ):
        if abs( y ) < 1e-14:
            return self.get_value_zero_displacement_with_scaling()

        result = self.get_value_hd( y )
        if result.convergence_flag:
            return result

        target_y = 0.0
        if y < self.y1:
            target_y = self.y1
        elif y > self.y2:
            target_y = self.y2

        if target_y > 0:
            T = ( target_y / y ) ** ( 2.0 * self.alpha ) * self.t
            scale = target_y / y
            result = self.get_value_plain_hd(
                target_y,
                T,
                ( self.t / T ) ** ( 1.0 - 1.0 / self.alpha ) * self.D_o
            )
            return FFPESolverResult(
                result.value * scale ** self.d,
                result.convergence_flag,
                result.value_2_difference
            )

        return result

    def get_value( self, y ):
        if self.D_f < 1e-14:
            return self.get_value_no_fractional_diffusion( y )
        if self.d == 1:
            return self.get_value_with_scaling_1d( float( y ) )
        return self.get_value_with_scaling_hd( float( y ) )

    def get_values( self, y_values ):
        results = []
        for y in y_values:
            results.append( self.get_value( float( y ) ) )
        return results

    def get_f( self, displacement, D_o, t ):
        def f( r ):
            bessel_values = evaluate_besselj_for_dimension( self.d, r * float( displacement ) )
            return (
                ( r / ( 2.0 * math.pi ) ) ** self.d_h
                * bessel_values
                * torch.exp( - float( D_o ) * float( t ) * r ** 2 )
            )
        return f

    def get_p_hat( self, displacement, D_o, D_f, alpha, t ):
        def p_hat( r ):
            bessel_values = evaluate_besselj_for_dimension( self.d, r * float( displacement ) )
            return (
                ( r / ( 2.0 * math.pi ) ) ** self.d_h
                * bessel_values
                * torch.exp( - float( D_o ) * float( t ) * r ** 2 )
                * torch.exp( - float( D_f ) * float( t ) * r ** ( 2.0 * float( alpha ) ) )
            )
        return p_hat

    def get_g( self, D_o, D_f, alpha, t ):
        def g( r ):
            return (
                ( r / ( 2.0 * math.pi ) ) ** self.d_h
                * torch.exp( - float( D_o ) * float( t ) * r ** 2 )
                * torch.exp( - float( D_f ) * float( t ) * r ** ( 2.0 * float( alpha ) ) )
            )
        return g

    def get_g_complement( self, displacement ):
        def g_complement( r ):
            return evaluate_besselj_for_dimension( self.d, r * float( displacement ) )
        return g_complement

    def get_f_zero_displacement( self, D_o, t ):
        def f( r ):
            return (
                ( r / ( 2.0 * math.pi ) ) ** self.d_m
                * torch.exp( - float( D_o ) * float( t ) * r ** 2 )
            )
        return f

    def get_p_hat_zero_displacement( self, D_o, D_f, alpha, t ):
        def p_hat( r ):
            return (
                ( r / ( 2.0 * math.pi ) ) ** self.d_m
                * torch.exp( - float( D_o ) * float( t ) * r ** 2 )
                * torch.exp( - float( D_f ) * float( t ) * r ** ( 2.0 * float( alpha ) ) )
            )
        return p_hat

    @staticmethod
    def get_f_1d( displacement, D_o, t ):
        def f( r ):
            return torch.cos( r * float( displacement ) ) * torch.exp( - float( D_o ) * float( t ) * r ** 2 )
        return f

    @staticmethod
    def get_p_hat_1d( displacement, D_o, D_f, alpha, t ):
        def p_hat( r ):
            return (
                torch.cos( r * float( displacement ) )
                * torch.exp( - float( D_o ) * float( t ) * r ** 2 )
                * torch.exp( - float( D_f ) * float( t ) * r ** ( 2.0 * float( alpha ) ) )
            )
        return p_hat

    @staticmethod
    def get_g_1d( D_o, D_f, alpha, t ):
        def g( r ):
            return (
                torch.exp( - float( D_o ) * float( t ) * r ** 2 )
                * torch.exp( - float( D_f ) * float( t ) * r ** ( 2.0 * float( alpha ) ) )
            )
        return g

    @staticmethod
    def get_g_complement_1d( displacement ):
        def g_complement( r ):
            return torch.cos( r * float( displacement ) )
        return g_complement
