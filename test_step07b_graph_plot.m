close all;
clear;
clc;

addpath( 'chebfun-master' );
addpath( 'windowing_functions' );

D_o = 1;
alpha = 0.5;
D_f = 8;
delta_t = 0.025;

d = 2;

y_limit = 2;
y_number = 51;
y = linspace( 0, y_limit, y_number );

t_limit = 0.2;
t_number = 51;
t = linspace( 0, t_limit, t_number );
t = t( 2 : end );

[ Y, T ] = meshgrid( y, t );

pv = zeros( size( Y ) );

tic;
for i = 1 : length( t )
    FFPESolver = C_FFPESolverV1M( d, alpha, D_o, D_f, t( i ) );
    FFPESolver.general_initialization();
    for j = 1 : length( y )
        pv( i, j ) = FFPESolver.get_value( y( j ) );
    end
end
toc;

set( gcf, 'Position', [ 40, 0, 1600, 1200 ] );
surf( Y, T, pv );
set( gca, 'ZScale', 'log' );
