close all;
clear;
clc;

addpath( 'chebfun-master' );
addpath( 'windowing_functions' );

x0 = 0;
b = 0;
D_o = 0;
p = 3;
q = 23;
alpha = p / q;
D_f = 8;

d = 1;

TEST_TYPE = 1;

if TEST_TYPE == 1
    % d = 5;
    % y = 0.04;
    % delta_t = 0.036;

    % delta_t = 0.004;
    delta_t = 0.04;
    y = 1.92;

    FFPESolver = C_FFPESolverV1M( d, alpha, D_o, D_f, delta_t );
    FFPESolver.general_initialization();
    a = FFPESolver.get_value( y );
end

% HA = C_FFPEHalfAlpha( D_o, D_f );
% HA = C_FFPEOneThirdAlpha( D_o, D_f );
% HA = C_FFPETwoThirdsAlpha( D_o, D_f );
HA = C_FFPERationalAlpha( p, q, D_o, D_f );
digits( 64 );

if TEST_TYPE == 2
    y_limit = 2;
    y_number = 51;
    y = linspace( 0, y_limit, y_number );

    t_limit = 0.2;
    % t_limit = 0.2 * 10;
    t_number = 51;
    t = linspace( 0, t_limit, t_number );
    t = t( 2 : end );

    [ Y, T ] = meshgrid( y, t );

    p = vpa( zeros( size( Y ) ) );
    pv = zeros( size( Y ) );

    tic;
    for i = 1 : length( t )
        p( i, : ) = HA.get_value( Y( i, : ), t( i ), d );
    end
    toc;

    tic;
    for i = 1 : length( t )
        FFPESolver = C_FFPESolverV1M( d, alpha, D_o, D_f, t( i ) );
        FFPESolver.general_initialization();
        for j = 1 : length( y )
            pv( i, j ) = FFPESolver.get_value( y( j ) );
        end
    end
    toc;

    set( gcf, 'Position', [ 0, 0, 2560, 1100 ] );
    sgtitle( sprintf( 'd = %02d, D_o = %.4f, D_f = %.4f, alpha = %.4f', d, D_o, D_f, alpha ) );

    subplot( 1, 2, 1 );
    surf( Y, T, double( pv ) );
    set( gca, 'ZScale', 'log' );
    title( 'Approximation Solution' );

    pd = abs( p - pv ) ./ p;
    subplot( 1, 2, 2 );
    surf( Y, T, double( pd ) );
    set( gca, 'ZScale', 'log' );
    title( 'Relative Error' );
end

if TEST_TYPE == 1
    b = HA.get_value( y, delta_t, d );
    
    b = vpa( b, 32 );
    fprintf( '%.9e\n', abs( a - b ) / abs( b ) );
end
