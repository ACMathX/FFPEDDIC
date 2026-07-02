close all;
clear;
clc;

config = F_load_json_config( 'configs/test_extra01_special_case_plot.json' );

D_o = config.D_o;
D_f = config.D_f;
d = config.d;

HA = C_FFPEHalfAlpha( D_o, D_f );
digits( config.digits );

% y_limit = 0.5;
y_limit = config.y_limit;
y_number = config.y_number;
y = linspace( 0, y_limit, y_number );

% t_limit = 0.1;
t_limit = config.t_limit;
t_number = config.t_number;
t = linspace( 0, t_limit, t_number );
t = t( 2 : end );

[ Y, T ] = meshgrid( y, t );

solution_values = vpa( zeros( size( Y ) ) );

tic;
for i = 1 : length( t )
    solution_values( i, 1 : end ) = HA.get_value( Y( i, 1 : end ), t( i ), d );
end
toc;

set( gcf, 'Position', [ 40, 0, 1600, 1200 ] );
surf( Y, T, double( solution_values ) );
set( gca, 'ZScale', 'log' );
