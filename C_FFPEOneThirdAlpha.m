classdef C_FFPEOneThirdAlpha < handle
    properties
        D_o ( 1, 1 ) double { mustBeNonnegative } = 0 % coefficient: ordinary diffusion
        D_f ( 1, 1 ) double { mustBeNonnegative } = 1 % coefficient: fractional diffusion
    end

    methods
        function [ self ] = C_FFPEOneThirdAlpha( D_o, D_f )
            if nargin > 0
                self.D_o = D_o;
            end
            if nargin > 1
                self.D_f = D_f;
            end
        end

        function [ value ] = get_value( self, y, t, d )
            arguments
                self ( 1, 1 ) C_FFPEOneThirdAlpha
                y    ( :, : ) double              { mustBeNonnegative }
                t    ( 1, 1 ) double              { mustBePositive }
                d    ( 1, 1 ) double              { mustBePositive, mustBeInteger }
            end
            if self.D_o == 0
                value = self.get_value_zero_D_o( y, t, d );
            else
                error( 'No implementation!' );
            end
        end

        function [ value ] = get_value_zero_D_o( self, y, t, d )
            arguments
                self ( 1, 1 ) C_FFPEOneThirdAlpha
                y    ( :, : ) double              { mustBeNonnegative }
                t    ( 1, 1 ) double              { mustBePositive }
                d    ( 1, 1 ) double              { mustBePositive, mustBeInteger }
            end
            tau = vpa( self.D_f .* t );
            y = vpa( y );
            d = vpa( d );
            d_h = vpa( d / 2 );
            p = vpa( pi );
            z = - ( y .^ 2 ) ./ ( tau .^ 3 ) .* ( 3 ^ 3 ) ./ ( 2 ^ 2 );
            a = [ d_h + 1 / 3, d_h + 2 / 3 ];
            coefficient = gamma( d_h * 3 ) ./ gamma( d_h ) .* 3 ./ ( 2 .^ d ) ./ ( p .^ d_h ) ./ ( tau .^ ( d_h * 3 ) );
            value = hypergeom( a, [], z ) .* coefficient;
        end
    end
end
