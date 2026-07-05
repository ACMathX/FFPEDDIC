classdef C_FFPERationalAlpha < handle
    properties
        p   ( 1, 1 ) double { mustBePositive, mustBeInteger } = 1
        q   ( 1, 1 ) double { mustBePositive, mustBeInteger } = 2
        D_o ( 1, 1 ) double { mustBeNonnegative }             = 0 % coefficient: ordinary diffusion
        D_f ( 1, 1 ) double { mustBeNonnegative }             = 1 % coefficient: fractional diffusion
    end

    methods
        function [ self ] = C_FFPERationalAlpha( p, q, D_o, D_f )
            if nargin < 1
                p = self.p;
            end
            if nargin < 2
                q = self.q;
            end
            if gcd( p, q ) ~= 1
                error( 'C_FFPERationalAlpha:InvalidRationalAlpha', 'p and q must be coprime.' );
            end
            if p >= q
                error( 'C_FFPERationalAlpha:InvalidRationalAlpha', 'p / q must be less than 1.' );
            end
            self.p = p;
            self.q = q;
            if nargin > 2
                self.D_o = D_o;
            end
            if nargin > 3
                self.D_f = D_f;
            end
        end

        function [ value ] = get_value( self, y, t, d )
            arguments
                self ( 1, 1 ) C_FFPERationalAlpha
                y    ( :, : ) double              { mustBeNonnegative }
                t    ( 1, 1 ) double              { mustBePositive }
                d    ( 1, 1 ) double              { mustBePositive, mustBeInteger }
            end
            if self.D_o == 0
                value = self.get_value_zero_D_o( y, t, d );
            else
                error( 'C_FFPERationalAlpha:UnsupportedOrdinaryDiffusion', 'Positive D_o is not implemented for rational alpha special cases.' );
            end
        end

        function [ value ] = get_value_zero_D_o( self, y, t, d )
            arguments
                self ( 1, 1 ) C_FFPERationalAlpha
                y    ( :, : ) double              { mustBeNonnegative }
                t    ( 1, 1 ) double              { mustBePositive }
                d    ( 1, 1 ) double              { mustBePositive, mustBeInteger }
            end
            tau = vpa( self.D_f .* t );
            y = vpa( y );
            d_high = vpa( d );
            d_h_high = vpa( d_high / 2 );
            p_high = vpa( self.p );
            q_high = vpa( self.q );
            r_high = p_high / q_high;
            pi_high = vpa( pi );
            z = - ( y ./ 2 ) .^ 2 ./ ( tau .^ ( 1 ./ r_high ) );
            coefficient = 1 ./ ( tau .^ ( d_h_high / r_high ) .* ( 2 * pi_high ) .^ d_high );
            c = ( z ./ ( p_high .^ 2 ) ) .^ p_high .* q_high .^ q_high;
            value = vpa( 0 );
            factorial_value = vpa( 1 );
            for i = 0 : self.p - 1
                if i > 1
                    factorial_value = factorial_value .* i;
                end
                i_high = vpa( i );
                f = self.get_n_sphere_surface_area( d + 2 * i - 1 ) .* gamma( ( d_h_high + i ) ./ r_high + 1 ) .* ( ( z ./ pi_high ) .^ i_high );
                f = f ./ ( factorial_value .* ( d_high + 2 * i ) );

                a = vpa( 1 : ( self.q - 1 ) ) ./ q_high + ( d_h_high + i_high ) ./ p_high;
                b1 = [ vpa( 0 : ( self.p - i - 2 ) ) + i_high + 1, vpa( ( self.p - i ) : ( self.p - 1 ) ) + i_high + 1 ] ./ p_high;
                b2 = ( vpa( 1 : ( self.p - 1 ) ) + d_h_high + i_high ) ./ p_high;
                b = [ b1, b2 ];
                value = value + hypergeom( a, b, c ) .* f;
            end
            value = value .* coefficient;
        end
    end

    methods ( Static )
        function [ S ] = get_n_sphere_surface_area( n )
            half_n_plus_1 = ( vpa( n ) + 1 ) / 2;
            p = vpa( pi );
            S = 2 * p .^ half_n_plus_1 / gamma( half_n_plus_1 );
        end
    end
end
