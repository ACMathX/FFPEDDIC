close all;
clear;
clc;

addpath( 'chebfun-master' );
addpath( 'windowing_functions' );

config = F_load_json_config( 'configs/test_step01_accuracy_verification.json' );

D_o = config.D_o;
alpha_numerator = config.alpha_numerator;
alpha_denominator = config.alpha_denominator;
alpha = alpha_numerator / alpha_denominator;
D_f = config.D_f;
d = config.d;

TEST_TYPE = config.test_type;

if TEST_TYPE == 1
    % d = 5;
    % y = 0.04;
    % delta_t = 0.036;

    % delta_t = 0.004;
    delta_t = config.delta_t;
    y = config.y;

    FFPESolver = C_FFPESolver( d, alpha, D_o, D_f, delta_t );
    FFPESolver.general_initialization();
    approximate_value = FFPESolver.get_value( y );
end

% HA = C_FFPEHalfAlpha( D_o, D_f );
% HA = C_FFPEOneThirdAlpha( D_o, D_f );
% HA = C_FFPETwoThirdsAlpha( D_o, D_f );
HA = C_FFPERationalAlpha( alpha_numerator, alpha_denominator, D_o, D_f );
digits( config.digits );

if TEST_TYPE == 2
    y_limit = config.y_limit;
    y_number = config.y_number;
    y = linspace( 0, y_limit, y_number );

    t_limit = config.t_limit;
    % t_limit = 0.2 * 10;
    t_number = config.t_number;
    t = linspace( 0, t_limit, t_number );
    t = t( 2 : end );

    [ Y, T ] = meshgrid( y, t );

    reference_solution = vpa( zeros( size( Y ) ) );
    approximate_solution = zeros( size( Y ) );

    tic;
    for i = 1 : length( t )
        reference_solution( i, : ) = HA.get_value( Y( i, : ), t( i ), d );
    end
    toc;

    tic;
    for i = 1 : length( t )
        FFPESolver = C_FFPESolver( d, alpha, D_o, D_f, t( i ) );
        FFPESolver.general_initialization();
        for j = 1 : length( y )
            approximate_solution( i, j ) = FFPESolver.get_value( y( j ) );
        end
    end
    toc;

    set( gcf, 'Position', [ 0, 0, 2560, 1100 ] );
    sgtitle( sprintf( 'd = %02d, D_o = %.4f, D_f = %.4f, alpha = %.4f', d, D_o, D_f, alpha ) );

    subplot( 1, 2, 1 );
    surf( Y, T, double( approximate_solution ) );
    set( gca, 'ZScale', 'log' );
    title( 'Approximation Solution' );

    relative_difference = abs( reference_solution - approximate_solution ) ./ reference_solution;
    subplot( 1, 2, 2 );
    surf( Y, T, double( relative_difference ) );
    set( gca, 'ZScale', 'log' );
    title( 'Relative Error' );
end

if TEST_TYPE == 1
    reference_value = HA.get_value( y, delta_t, d );
    
    reference_value = vpa( reference_value, 32 );
    fprintf( '%.9e\n', abs( approximate_value - reference_value ) / abs( reference_value ) );
end
