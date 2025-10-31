close all;
clear;
clc;

D_o = 1;
D_f = 8;

HA = C_FFPEHalfAlpha( D_o, D_f );
digits( 64 );

% y_limit = 0.5;
y_limit = 2;
y_number = 51;
y = linspace( 0, y_limit, y_number );

% t_limit = 0.1;
t_limit = 0.2;
t_number = 51;
t = linspace( 0, t_limit, t_number );
t = t( 2 : end );

[ Y, T ] = meshgrid( y, t );

p = vpa( zeros( size( Y ) ) );

d = 1;
tic;
for i = 1 : length( t )
    p( i, 1 : end ) = HA.get_value( Y( i, 1 : end ), t( i ), d );
end
toc;

set( gcf, 'Position', [ 40, 0, 1600, 1200 ] );
surf( Y, T, double( p ) );
set( gca, 'ZScale', 'log' );
