classdef C_FFPESolverV1M < handle
    properties
        d     ( 1, 1 ) double { mustBeInteger }     = 1      % dimension
        alpha ( 1, 1 ) double { mustBePositive }    = 0.5    % fractional exponent
        D_o   ( 1, 1 ) double { mustBeNonnegative } = 0      % coefficient of the ordinary diffusion
        D_f   ( 1, 1 ) double { mustBeNonnegative } = 1      % coefficient of the fractional diffusion
        t     ( 1, 1 ) double { mustBePositive }    = 0.1    % time
        d_r   ( 1, 1 ) double                                % d reduced: (d - 2) / 2
        d_h   ( 1, 1 ) double                                % d half: d / 2
        d_m   ( 1, 1 ) double                                % d minus: d - 1
        coef  ( 1, 1 ) double                                % coefficient for zero displacement: S_{d - 1} / (2 \pi)

        L     ( 1, 1 ) double { mustBePositive }    = 1
        s1    ( :, 1 ) double                                % near origin quadrature points
        w1    ( 1, : ) double                                % near origin quadrature weights
        y1    ( 1, 1 ) double                       = pi / 2
        y2    ( 1, 1 ) double                       = pi / 2
        t1    ( 1, 1 ) double                       = 1 / 10

        s2    ( :, 1 ) cell                                  % windowing function part quadrature points
        w2    ( :, 1 ) cell                                  % windowing function part quadrature weights
        v2    ( :, 1 ) cell                                  % windowing function part value at quadrature points
        g2    ( :, 1 ) cell                                  % windowing function part partial value at quadrature points

        M_ini ( 1, 1 ) double { mustBePositive }    = 80     % M lower bound
        gamma ( 1, 1 ) double { mustBePositive }    = 0.5    % windowing function gamma, which describes how fast is the transition
        d_tol ( 1, 1 ) double { mustBePositive }    = 1e-14  % difference tolerance, stop criterion
        M_lim ( 1, 1 ) double { mustBePositive }    = 5121   % M upper bound
    end

    methods
        function [ self ] = C_FFPESolverV1M( d, alpha, D_o, D_f, t )
            self.d = d;
            self.alpha = alpha;
            self.D_o = D_o;
            self.D_f = D_f;
            self.t = t;

            if self.d > 1
                self.d_r = ( self.d - 2 ) / 2;
            else
                self.d_r = 0;
            end
            self.d_h = self.d / 2;
            self.d_m = self.d - 1;
            self.coef = self.compute_coefficient( self.d_m );
        end
    end

    methods % initializations
        function [ ] = windowing_function_initialization( self )
            M = self.M_ini;
            cell_length = 0;
            while M < self.M_lim
                cell_length = cell_length + 1;
                M = M * 2;
            end

            self.s2 = cell( cell_length, 1 );
            self.w2 = cell( cell_length, 1 );
            self.v2 = cell( cell_length, 1 );

            M = self.M_ini;
            cell_index = 1;
            while M < self.M_lim
                windowing_function = C_TestWindowingFunction02( M, self.gamma );
                N = 50 * M;
                [ self.s2{ cell_index }, self.w2{ cell_index } ] = legpts( N, [ self.L, M ] );
                self.v2{ cell_index } = windowing_function.get_value( self.s2{ cell_index } );
                cell_index = cell_index + 1;
                M = M * 2;
            end
        end

        function [ ] = quadrature_initialization( self, n, L, eps )
            if nargin < 2
                n = 16;
            end
            if nargin < 3
                L = 1;
            end
            if nargin < 4
                eps = 1e-14;
            end
            self.L = L;
            FQ = C_FractionalQuadrature( self.alpha, self.D_f * self.t, self.L );

            [ self.s1, self.w1 ] = FQ.get_weights_by_exactness( n, eps );
        end

        function [ ] = update_g2( self )
            if self.d == 1
                g = self.get_g_1d( self.D_o, self.D_f, self.alpha, self.t );
            else
                g = self.get_g( self.D_o, self.D_f, self.alpha, self.t );
            end
            self.g2 = cell( size( self.s2 ) );

            cell_index = 1;
            M = self.M_ini;
            while M < self.M_lim
                self.g2{ cell_index } = g( self.s2{ cell_index } ) .* self.v2{ cell_index };
                cell_index = cell_index + 1;
                M = M * 2;
            end
        end

        function [ ] = general_initialization( self )
            self.windowing_function_initialization();
            self.quadrature_initialization( 16, self.L );
            self.update_g2();
        end
    end

    methods
        function [ value, convergence_flag, value_2_difference ] = get_value_1d( self, y )
            f = self.get_f_1d( y, self.D_o, self.t );
            g_complement = self.get_g_complement_1d( y );

            convergence_flag = false;
            last_value_2 = Inf;
            current_value_2 = 0;
            M = self.M_ini;
            % directly assume convergence
            cell_index = 1;
            while M < self.M_lim
                s2_ = self.s2{ cell_index };
                w2_ = self.w2{ cell_index };
                g2_ = self.g2{ cell_index };
                current_value_2 = w2_ * ( g_complement( s2_ ) .* g2_ );
                value_2_difference = abs( last_value_2 - current_value_2 );
                if value_2_difference < self.d_tol
                    convergence_flag = true;
                    break;
                end
                last_value_2 = current_value_2;
                cell_index = cell_index + 1;
                M = M * 2;
            end
            % fprintf( 'M: %d\n', M );
            % fprintf( 'part_2_diff: %.4e\n', value_2_difference );
            value_2 = current_value_2;

            value_1 = self.w1 * f( self.s1 );

            value = value_2 + value_1;
            value = value ./ pi;
        end

        function [ value, convergence_flag, value_2_difference ] = get_value_hd( self, y )
            % y > 0
            f = self.get_f( y, self.D_o, self.t );
            g_complement = self.get_g_complement( y );

            convergence_flag = false;
            last_value_2 = Inf;
            current_value_2 = 0;
            M = self.M_ini;
            % directly assume convergence
            cell_index = 1;
            while M < self.M_lim
                s2_ = self.s2{ cell_index };
                w2_ = self.w2{ cell_index };
                g2_ = self.g2{ cell_index };
                current_value_2 = w2_ * ( g_complement( s2_ ) .* g2_ );
                value_2_difference = abs( last_value_2 - current_value_2 );
                if value_2_difference < self.d_tol
                    convergence_flag = true;
                    break;
                end
                last_value_2 = current_value_2;
                cell_index = cell_index + 1;
                M = M * 2;
            end
            % fprintf( 'M: %d\n', M );
            % fprintf( 'part_2_diff: %.4e\n', value_2_difference );
            value_2 = current_value_2;

            value_1 = self.w1 * f( self.s1 );

            value = value_2 + value_1;
            value = value ./ ( y .^ self.d_r );
        end

        function [ value, convergence_flag, value_2_difference ] = get_value_plain_1d( self, y, t, D_o )
            f = self.get_f_1d( y, D_o, t );
            p_hat = self.get_p_hat_1d( y, D_o, self.D_f, self.alpha, t );

            convergence_flag = false;
            last_value_2 = Inf;
            current_value_2 = 0;
            M = self.M_ini;
            % directly assume convergence
            cell_index = 1;
            while M < self.M_lim
                s2_ = self.s2{ cell_index };
                w2_ = self.w2{ cell_index };
                v2_ = self.v2{ cell_index };
                current_value_2 = w2_ * ( p_hat( s2_ ) .* v2_ );
                value_2_difference = abs( last_value_2 - current_value_2 );
                if value_2_difference < self.d_tol
                    convergence_flag = true;
                    break;
                end
                last_value_2 = current_value_2;
                cell_index = cell_index + 1;
                M = M * 2;
            end
            % fprintf( 'M: %d\n', M );
            % fprintf( 'part_2_diff: %.4e\n', value_2_difference );
            value_2 = current_value_2;

            FQ = C_FractionalQuadrature( self.alpha, self.D_f * t, self.L );
            value_1 = FQ.get_value( 16, f );

            value = value_2 + value_1;
            value = value ./ pi;
        end

        function [ value, convergence_flag, value_2_difference ] = get_value_plain_hd( self, y, t, D_o )
            f = self.get_f( y, D_o, t );
            p_hat = self.get_p_hat( y, D_o, self.D_f, self.alpha, t );

            convergence_flag = false;
            last_value_2 = Inf;
            current_value_2 = 0;
            M = self.M_ini;
            % directly assume convergence
            cell_index = 1;
            while M < self.M_lim
                s2_ = self.s2{ cell_index };
                w2_ = self.w2{ cell_index };
                v2_ = self.v2{ cell_index };
                current_value_2 = w2_ * ( p_hat( s2_ ) .* v2_ );
                value_2_difference = abs( last_value_2 - current_value_2 );
                if value_2_difference < self.d_tol
                    convergence_flag = true;
                    break;
                end
                last_value_2 = current_value_2;
                cell_index = cell_index + 1;
                M = M * 2;
            end
            % fprintf( 'M: %d\n', M );
            % fprintf( 'part_2_diff: %.4e\n', value_2_difference );
            value_2 = current_value_2;

            FQ = C_FractionalQuadrature( self.alpha, self.D_f * t, self.L );
            value_1 = FQ.get_value( 16, f );

            value = value_2 + value_1;
            value = value ./ ( y .^ self.d_r );
        end
    end

    methods
        function [ value, convergence_flag, value_2_difference ] = get_value_zero_displacement_plain( self, t, D_o )
            f = self.get_f_zero_displacement( D_o, t );
            p_hat = self.get_p_hat_zero_displacement( D_o, self.D_f, self.alpha, t );

            convergence_flag = false;
            last_value_2 = Inf;
            current_value_2 = 0;
            M = self.M_ini;
            % directly assume convergence
            cell_index = 1;
            while M < self.M_lim
                s2_ = self.s2{ cell_index };
                w2_ = self.w2{ cell_index };
                v2_ = self.v2{ cell_index };
                current_value_2 = w2_ * ( p_hat( s2_ ) .* v2_ );
                value_2_difference = abs( last_value_2 - current_value_2 );
                if value_2_difference < self.d_tol
                    convergence_flag = true;
                    break;
                end
                last_value_2 = current_value_2;
                cell_index = cell_index + 1;
                M = M * 2;
            end
            % fprintf( 'M: %d\n', M );
            % fprintf( 'part_2_diff: %.4e\n', value_2_difference );
            value_2 = current_value_2;

            FQ = C_FractionalQuadrature( self.alpha, self.D_f * t, self.L );
            value_1 = FQ.get_value( 16, f );

            value = value_2 + value_1;
            value = value .* self.coef;
        end

        function [ value, convergence_flag, value_2_difference ] = get_value_zero_displacement( self )
            [ value, convergence_flag, value_2_difference ] = self.get_value_zero_displacement_plain( self.t, self.D_o );
        end

        function [ value, convergence_flag, value_2_difference ] = get_value_zero_displacement_with_scaling( self )
            if abs( self.D_o ) < 1e-12
                convergence_flag = true;
                value_2_difference = 0;
                value = ( self.D_f .* self.t ) .^ ( - self.d ./ 2 ./ self.alpha ) .* gamma( self.d ./ 2 ./ self.alpha + 1 );
                value = value ./ ( 2 .* pi ) .^ self.d ./ self.d .* 2 .* pi .^ ( self.d ./ 2 ) ./ gamma( self.d ./ 2 );
                return;
            end
            [ value, convergence_flag, value_2_difference ] = self.get_value_zero_displacement();
            if ~ convergence_flag
                target_D_o = 10;
                T = ( self.D_o ./ target_D_o ) .^ ( 1 ./ ( 1 - 1 ./ self.alpha ) ) .* self.t;
                scale = ( self.t ./ T ) .^ ( - 1 ./ ( 2 .* self.alpha ) );
                [ value, convergence_flag, value_2_difference ] = self.get_value_zero_displacement_plain( T, target_D_o );
                value = value .* scale .^ self.d;
            end
        end
    end

    methods
        function [ value ] = get_value_no_fractional_diffusion( self, y )
            denominator1 = 4 * self.D_o * self.t;
            denominator2 = ( denominator1 * pi ) .^ self.d_h;
            value = exp( - y .^ 2 ./ denominator1 ) ./ denominator2;
        end
    end

    methods
        function [ value, convergence_flag, value_2_difference ] = get_value_with_scaling_1d( self, y )
            if abs( y ) < 1e-14
                [ value, convergence_flag, value_2_difference ] = self.get_value_zero_displacement_with_scaling();
                return;
            end
            [ value, convergence_flag, value_2_difference ] = self.get_value_1d( y );
            if ~ convergence_flag
                target_y = 0;
                if y < self.y1
                    target_y = self.y1;
                elseif y > self.y2
                    target_y = self.y2;
                end
                if target_y > 0
                    T = ( target_y / y ) .^ ( 2 * self.alpha ) * self.t;
                    scale = target_y / y; % ( t / T )^{ - 1 / 2 alpha }
                    [ value, convergence_flag, value_2_difference ] = self.get_value_plain_1d( target_y, T, ( self.t / T ) .^ ( 1 - 1 / self.alpha ) .* self.D_o );
                    value = value .* scale;
                end
                % if ~ convergence_flag
                %     if self.t < self.t1
                %         T = self.t1;
                %         scale = ( self.t / T ) .^ ( - 1 / ( 2 * self.alpha ) );
                %         target_y = scale * y;
                %         [ value, convergence_flag, value_2_difference ] = self.get_value_plain_1d( target_y, T, ( self.t / T ) .^ ( 1 - 1 / self.alpha ) .* self.D_o );
                %         value = value .* scale;
                %     end
                % end
            end
        end

        function [ value, convergence_flag, value_2_difference ] = get_value_with_scaling_hd( self, y )
            if abs( y ) < 1e-14
                [ value, convergence_flag, value_2_difference ] = self.get_value_zero_displacement_with_scaling();
                return;
            end
            [ value, convergence_flag, value_2_difference ] = self.get_value_hd( y );
            if ~ convergence_flag
                target_y = 0;
                if y < self.y1
                    target_y = self.y1;
                elseif y > self.y2
                    target_y = self.y2;
                end
                if target_y > 0
                    T = ( target_y / y ) .^ ( 2 * self.alpha ) * self.t;
                    scale = target_y / y; % ( t / T )^{ - 1 / 2 alpha }
                    [ value, convergence_flag, value_2_difference ] = self.get_value_plain_hd( target_y, T, ( self.t / T ) .^ ( 1 - 1 / self.alpha ) .* self.D_o );
                    value = value .* scale .^ self.d;
                end
                % if ~ convergence_flag
                %     if self.t < self.t1
                %         T = self.t1;
                %         scale = ( self.t / T ) .^ ( - 1 / ( 2 * self.alpha ) );
                %         target_y = scale * y;
                %         [ value, convergence_flag, value_2_difference ] = self.get_value_plain_hd( target_y, T, ( self.t / T ) .^ ( 1 - 1 / self.alpha ) .* self.D_o );
                %         value = value .* scale .^ self.d;
                %     end
                % end
            end
        end

        function [ value, convergence_flag ] = get_value( self, y )
            if self.D_f < 1e-14
                convergence_flag = true;
                value = self.get_value_no_fractional_diffusion( y );
                return;
            end
            if self.d == 1
                [ value, convergence_flag ] = self.get_value_with_scaling_1d( y );
            else
                [ value, convergence_flag ] = self.get_value_with_scaling_hd( y );
            end
        end
    end

    methods
        function [ f ] = get_f( self, displacement, D_o, t )
            f = @( r ) ( r ./ 2 ./ pi ) .^ self.d_h .* besselj( self.d_r, r .* displacement ) .* exp( - D_o .* t .* r .^ 2 );
        end

        function [ p_hat ] = get_p_hat( self, displacement, D_o, D_f, alpha, t )
            p_hat = @( r ) ( r ./ 2 ./ pi ) .^ self.d_h .* besselj( self.d_r, r .* displacement ) .* exp( - D_o .* t .* r .^ 2 ) .* exp( - D_f .* t .* r .^ ( 2 .* alpha ) );
        end

        function [ g ] = get_g( self, D_o, D_f, alpha, t )
            g = @( r ) ( r ./ 2 ./ pi ) .^ self.d_h .* exp( - D_o .* t .* r .^ 2 ) .* exp( - D_f .* t .* r .^ ( 2 .* alpha ) );
        end

        function [ g_complement ] = get_g_complement( self, displacement )
            g_complement = @( r ) besselj( self.d_r, r .* displacement );
        end
    end

    methods
        function [ f ] = get_f_zero_displacement( self, D_o, t )
            f = @( r ) ( r ./ 2 ./ pi ) .^ self.d_m .* exp( - D_o .* t .* r .^ 2 );
        end

        function [ p_hat ] = get_p_hat_zero_displacement( self, D_o, D_f, alpha, t )
            p_hat = @( r ) ( r ./ 2 ./ pi ) .^ self.d_m .* exp( - D_o .* t .* r .^ 2 ) .* exp( - D_f .* t .* r .^ ( 2 .* alpha ) );
        end
    end

    methods ( Static )
        function [ f ] = get_f_1d( displacement, D_o, t )
            f = @( r ) cos( r .* displacement ) .* exp( - D_o .* t .* r .^ 2 );
        end
        
        function [ p_hat ] = get_p_hat_1d( displacement, D_o, D_f, alpha, t )
            p_hat = @( r ) cos( r .* displacement ) .* exp( - D_o .* t .* r .^ 2 ) .* exp( - D_f .* t .* r .^ ( 2 .* alpha ) );
        end

        function [ g ] = get_g_1d( D_o, D_f, alpha, t )
            g = @( r ) exp( - D_o .* t .* r .^ 2 ) .* exp( - D_f .* t .* r .^ ( 2 .* alpha ) );
        end

        function [ g_complement ] = get_g_complement_1d( displacement )
            g_complement = @( r ) cos( r .* displacement );
        end

        function [ value ] = compute_coefficient( n )
            value = 2 .* ( pi .^ ( ( n + 1 ) ./ 2 ) ) ./ gamma( ( n + 1 ) ./ 2 );
            value = value ./ ( 2 .* pi );
        end
    end
end
