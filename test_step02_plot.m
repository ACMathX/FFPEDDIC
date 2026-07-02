close all;
clear;
clc;

addpath( 'chebfun-master' );
addpath( 'windowing_functions' );

config = F_load_json_config( 'configs/test_step02_plot.json' );

D_o = config.D_o;
alpha = config.alpha;
D_f = config.D_f;
d = config.d;

y_limit = config.y_limit;
y_number = config.y_number;
y = linspace( 0, y_limit, y_number );

t_limit = config.t_limit;
t_number = config.t_number;
t = linspace( 0, t_limit, t_number );
t = t( 2 : end );

[ Y, T ] = meshgrid( y, t );

solution_values = zeros( size( Y ) );

tic;
for i = 1 : length( t )
    FFPESolver = C_FFPESolver( d, alpha, D_o, D_f, t( i ) );
    FFPESolver.general_initialization();
    for j = 1 : length( y )
        solution_values( i, j ) = FFPESolver.get_value( y( j ) );
    end
end
toc;

set( gcf, 'Position', [ 40, 0, 1600, 1200 ] );
surf( Y, T, solution_values );
set( gca, 'ZScale', 'log' );
