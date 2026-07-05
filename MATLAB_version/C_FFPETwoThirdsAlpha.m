classdef C_FFPETwoThirdsAlpha < handle
    properties
        D_o ( 1, 1 ) double { mustBeNonnegative } = 0 % coefficient: ordinary diffusion
        D_f ( 1, 1 ) double { mustBeNonnegative } = 1 % coefficient: fractional diffusion
    end

    methods
        function [ self ] = C_FFPETwoThirdsAlpha( D_o, D_f )
            if nargin > 0
                self.D_o = D_o;
            end
            if nargin > 1
                self.D_f = D_f;
            end
        end

        function [ value ] = get_value( self, y, t, d )
            arguments
                self ( 1, 1 ) C_FFPETwoThirdsAlpha
                y    ( :, : ) double               { mustBeNonnegative }
                t    ( 1, 1 ) double               { mustBePositive }
                d    ( 1, 1 ) double               { mustBePositive, mustBeInteger }
            end
            if self.D_o == 0
                value = self.get_value_zero_D_o( y, t, d );
            else
                error( 'C_FFPETwoThirdsAlpha:UnsupportedOrdinaryDiffusion', 'Positive D_o is not implemented for alpha = 2 / 3 special cases.' );
            end
        end

        function [ value ] = get_value_zero_D_o( self, y, t, d )
            arguments
                self ( 1, 1 ) C_FFPETwoThirdsAlpha
                y    ( :, : ) double               { mustBeNonnegative }
                t    ( 1, 1 ) double               { mustBePositive }
                d    ( 1, 1 ) double               { mustBePositive, mustBeInteger }
            end
            tau = vpa( self.D_f .* t );
            y = vpa( y );
            d = vpa( d );
            d_4 = vpa( d / 4 );
            d_4_2 = vpa( ( d + 2 ) / 4 );
            p = vpa( pi );
            z = ( y .^ 4 ) ./ ( tau .^ 3 ) .* ( 3 ^ 3 ) ./ ( 4 ^ 4 );
            a1 = [ d_4 + 1 / 3, d_4 + 2 / 3 ];
            b1 = [ vpa( 1 ) / 2, d_4_2 ];
            a2 = [ d_4_2 + 1 / 3, d_4_2 + 2 / 3 ];
            b2 = [ vpa( 3 ) / 2, d_4 + 1 ];
            value_1 = hypergeom( a1, b1, z ) .* gamma( d_4 * 3 ) ./ gamma( d_4 * 2 );
            value_2 = hypergeom( a2, b2, z ) .* gamma( d_4_2 * 3 ) ./ gamma( d_4_2 * 2 ) .* ( y ./ 2 ) .^ 2 ./ ( tau .^ ( vpa( 3 ) / 2 ) );
            coefficient = 3 ./ ( 2 .^ ( d + 1 ) ) ./ ( p .^ ( d ./ 2 ) ) ./ ( tau .^ ( d_4 * 3 ) );
            value = ( value_1 - value_2 ) .* coefficient;
        end
    end
end
