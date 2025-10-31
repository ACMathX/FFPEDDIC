classdef C_FractionalQuadrature < handle
    properties
        alpha    (1, 1) double {mustBeNonnegative} = 0.5      % fractional power
        delta_t  (1, 1) double {mustBeNonnegative} = 1e-2     % time difference T - t
        L        (1, 1) double {mustBePositive}    = 1        % integration interval parameter
        interval (1, 2) double                     = [ 0, 1 ] % integration interval [ 0, L ]
    end
    
    methods
        function [ self ] = C_FractionalQuadrature( alpha, delta_t, L )
            self.alpha = alpha;
            self.delta_t = delta_t;
            self.L = L;
            self.interval = [ 0, self.L ];
        end

        function [ s, w ] = get_weights_by_exactness( self, n, eps )
            % suggest n <= 32
            if nargin < 3
                eps = 1e-14;
            end

            BETA = self.alpha * 2;

            [ s, ~ ] = legpts( n, self.interval );

            F = zeros( n, 1 );

            LP = C_LegendrePolynomial();
            y = 2 * s' / self.L - 1;
            P = arrayfun( @LP.get_polynomial, 0 : n - 1, 'UniformOutput', false );
            A = zeros( n );
            for k = 1 : n
                [ F( k ), ~ ] = quadgk( @( x ) exp( - abs( x ) .^ BETA * self.delta_t ) .* P{ k }( 2 * x / self.L - 1 ), 0, self.L, 'AbsTol', eps, 'RelTol', 100 * eps, 'MaxIntervalCount', 1000 );
                A( k, : ) = P{ k }( y );
            end

            w = ( A \ F )';
        end

        function [ s, w ] = get_weights_by_exactness_gradient( self, n, eps )
            % suggest n <= 32
            if nargin < 3
                eps = 1e-14;
            end

            BETA = self.alpha * 2;

            [ s, ~ ] = legpts( n, self.interval );

            F = zeros( n, 1 );

            LP = C_LegendrePolynomial();
            y = 2 * s' / self.L - 1;
            P = arrayfun( @LP.get_polynomial, 0 : n - 1, 'UniformOutput', false );
            A = zeros( n );
            for k = 1 : n
                [ F( k ), ~ ] = quadgk( @( x ) abs( x ) .^ BETA .* log( abs( x ) ) .* exp( - abs( x ) .^ BETA * self.delta_t ) .* P{ k }( 2 * x / self.L - 1 ), 0, self.L, 'AbsTol', eps, 'RelTol', 100 * eps, 'MaxIntervalCount', 1000 );
                A( k, : ) = P{ k }( y );
            end

            w = ( A \ F )';
        end

        function [ s, w ] = get_weights_by_exactness_gradient_modified( self, n, eps )
            % suggest n <= 32
            if nargin < 3
                eps = 1e-14;
            end

            BETA = self.alpha * 2;

            [ s, ~ ] = legpts( n, self.interval );

            F = zeros( n, 1 );

            LP = C_LegendrePolynomial();
            y = 2 * s' / self.L - 1;
            P = arrayfun( @LP.get_polynomial, 0 : n - 1, 'UniformOutput', false );
            A = zeros( n );
            for k = 1 : n
                [ F( k ), ~ ] = quadgk( @( x ) abs( x ) .^ BETA .* exp( - abs( x ) .^ BETA * self.delta_t ) .* P{ k }( 2 * x / self.L - 1 ), 0, self.L, 'AbsTol', eps, 'RelTol', 100 * eps, 'MaxIntervalCount', 1000 );
                A( k, : ) = P{ k }( y );
            end

            w = ( A \ F )';
        end
        
        function [ value ] = get_value( self, n, f )
            % need to check whether the expansion takes the advantages in
            % term of the time
            if self.delta_t <= 1e-3
                value = self.compute_by_expansion( n, f, 4 );
            elseif self.delta_t <= 1e-2
                value = self.compute_by_expansion( n, f, 6 );
            elseif self.delta_t <= 1e-1
                value = self.compute_by_expansion( n, f, 9 );
            elseif self.delta_t <= 1e0
                value = self.compute_by_expansion( n, f, 17 );
            else % self.delta_t > 1
                value = self.compute_by_exactness( n, f );
            end
        end

        function [ value ] = get_value_modified( self, n, f )
            if self.delta_t <= 1e-3
                value = self.compute_by_expansion_modified( n, f, 4 );
            elseif self.delta_t <= 1e-2
                value = self.compute_by_expansion_modified( n, f, 6 );
            elseif self.delta_t <= 1e-1
                value = self.compute_by_expansion_modified( n, f, 9 );
            elseif self.delta_t <= 1e0
                value = self.compute_by_expansion_modified( n, f, 17 );
            else % self.delta_t > 1
                value = self.compute_by_exactness_gradient_modified( n, f );
            end
        end
    end

    methods
        function [ value ] = compute_directly( self, f, eps )
            if nargin < 3
                eps = 1e-16;
            end
            BETA = self.alpha * 2;
            value = integral( @( x ) exp( - abs( x ) .^ BETA * self.delta_t ) .* f( x ), 0, self.L, 'AbsTol', eps, 'RelTol', eps );
        end

        function [ value ] = compute_directly_gradient( self, f, eps )
            if nargin < 3
                eps = 1e-16;
            end
            BETA = self.alpha * 2;
            value = integral( @( x ) abs( x ) .^ BETA .* log( abs( x ) ) .* exp( - abs( x ) .^ BETA * self.delta_t ) .* f( x ), 0, self.L, 'AbsTol', eps, 'RelTol', eps );
        end

        function [ value ] = compute_directly_gradient_modified( self, f, eps )
            if nargin < 3
                eps = 1e-16;
            end
            BETA = self.alpha * 2;
            value = integral( @( x ) abs( x ) .^ BETA .* exp( - abs( x ) .^ BETA * self.delta_t ) .* f( x ), 0, self.L, 'AbsTol', eps, 'RelTol', eps );
        end

        function [ value ] = compute_by_exactness( self, n, f )
            [ s, w ] = self.get_weights_by_exactness( n, 1e-14 );
            value = w * f( s );
        end

        function [ value ] = compute_by_exactness_gradient( self, n, f )
            [ s, w ] = self.get_weights_by_exactness_gradient( n, 1e-14 );
            value = w * f( s );
        end

        function [ value ] = compute_by_exactness_gradient_modified( self, n, f )
            [ s, w ] = self.get_weights_by_exactness_gradient_modified( n, 1e-14 );
            value = w * f( s );
        end

        function [ value ] = compute_by_expansion( self, n, f, number_of_term )
            % until number_of_term's derivative
            [ s, w ] = legpts( n, self.interval );
            value = w * f( s );
            BETA = self.alpha * 2;
            multiplier = 1;
            for k = 1 : number_of_term
                multiplier = - multiplier * self.delta_t / k;
                current_beta = k * BETA;
                if current_beta <= 5
                    [ s, w ] = jacpts( n, 0, current_beta, self.interval );
                    value = value + multiplier * w * f( s );
                else
                    [ s, w ] = legpts( n, self.interval );
                    value = value + multiplier * w * ( abs( s ) .^ current_beta .* f( s ) );
                end
            end
        end

        function [ value ] = compute_by_expansion_modified( self, n, f, number_of_term )
            % until number_of_term's derivative
            BETA = self.alpha * 2;

            [ s, w ] = jacpts( n, 0, BETA, self.interval );
            value = w * f( s );
            multiplier = 1;
            for k = 1 : number_of_term
                multiplier = - multiplier * self.delta_t / k;
                current_beta = ( k + 1 ) * BETA;
                if current_beta <= 5
                    [ s, w ] = jacpts( n, 0, current_beta, self.interval );
                    value = value + multiplier * w * f( s );
                else
                    [ s, w ] = legpts( n, self.interval );
                    value = value + multiplier * w * ( abs( s ) .^ current_beta .* f( s ) );
                end
            end
        end
    end
end
