classdef C_FFPEHalfAlpha < handle
    % Need "Faddeeva Package: complex error functions" (Version 1.5.0.0)
    properties
        D_o ( 1, 1 ) double { mustBeNonnegative } = 1 % coefficient: ordinary diffusion
        D_f ( 1, 1 ) double { mustBeNonnegative } = 1 % coefficient: fractional diffusion
    end

    methods
        function [ self ] = C_FFPEHalfAlpha( D_o, D_f )
            if nargin > 0
                self.D_o = D_o;
            end
            if nargin > 1
                self.D_f = D_f;
            end
        end

        function [ value ] = get_value( self, y, t, d )
            arguments
                self ( 1, 1 ) C_FFPEHalfAlpha
                y    ( :, : ) double          { mustBeNonnegative }
                t    ( 1, 1 ) double          { mustBePositive }
                d    ( 1, 1 ) double          { mustBePositive, mustBeInteger }
            end
            if self.D_o == 0
                value = self.get_value_zero_D_o( y, t, d );
            else
                positive_position = y > 0;
                zero_position = y == 0;
                value = vpa( zeros( size( y ) ) );
                value( zero_position ) = self.get_value_positive_D_o_zero_y( t, d );
                value( positive_position ) = self.get_value_positive_D_o( y( positive_position ), t, d );
            end
        end

        function [ value ] = get_value_positive_D_o( self, y, t, d )
            arguments
                self ( 1, 1 ) C_FFPEHalfAlpha
                y    ( :, : ) double          { mustBePositive }
                t    ( 1, 1 ) double          { mustBePositive }
                d    ( 1, 1 ) double          { mustBePositive, mustBeInteger, mustBeOdd }
            end
            component_base = 2 * sqrt( self.D_o * t );
            P_component = ( self.D_f * t - y * 1j ) / component_base;
            M_component = ( self.D_f * t + y * 1j ) / component_base;
            y = vpa( y );

            % Small d scenarios
            if d == 1
                % value = vpa( Faddeeva_erfcx( P_component ) + Faddeeva_erfcx( M_component ) ) / ( 2 * sqrt( 4 * pi * self.A * t ) );
                value = ( F_erfcx( P_component ) + F_erfcx( M_component ) ) / ( 2 * sqrt( 4 * pi * self.D_o * t ) );
                return;
            % elseif d == 3 % can be deleted
            %     value = ( - Faddeeva_erfcx( P_component ) .* ( self.C * t - y * 1j ) + Faddeeva_erfcx( M_component ) .* ( self.C * t + y * 1j ) ) ./ ( y * 1j ) / ( 2 * sqrt( 4 * pi * self.A * t ) ^ 3 );
            %     return;
            end

            % Get all K^{+}_{q} and K^{-}_{q}
            q_limit = ( d + 1 ) / 2;
            K_recursion_base = vpa( 2 * self.D_o * t );
            KP_recursion = - ( self.D_f * t - y * 1j ) / K_recursion_base;
            KM_recursion = - ( self.D_f * t + y * 1j ) / K_recursion_base;

            KP = vpa( zeros( size( y, 1 ), size( y, 2 ), q_limit ) );
            KM = vpa( zeros( size( y, 1 ), size( y, 2 ), q_limit ) );

            % KP( :, :, 1 ) = sqrt( vpa( pi ) ) / component_base * Faddeeva_erfcx( P_component ); % K^{+}_{0}
            KP( :, :, 1 ) = sqrt( vpa( pi ) ) / component_base * F_erfcx( P_component ); % K^{+}_{0}
            KP( :, :, 2 ) = 1 / K_recursion_base + KP_recursion .* KP( :, :, 1 ); % K^{+}_{1}

            % KM( :, :, 1 ) = sqrt( vpa( pi ) ) / component_base * Faddeeva_erfcx( M_component ); % K^{-}_{0}
            KM( :, :, 1 ) = sqrt( vpa( pi ) ) / component_base * F_erfcx( M_component ); % K^{-}_{0}
            KM( :, :, 2 ) = 1 / K_recursion_base + KM_recursion .* KM( :, :, 1 ); % K^{-}_{1}

            for q = 3 : q_limit
                KP( :, :, q ) = KP_recursion .* KP( :, :, q - 1 ) + ( q - 2 ) / K_recursion_base * KP( :, :, q - 2 );
                KM( :, :, q ) = KM_recursion .* KM( :, :, q - 1 ) + ( q - 2 ) / K_recursion_base * KM( :, :, q - 2 );
            end

            memoize_cache = dictionary( { [] }, { [] } );

            function [ value ] = T( p, q )
                if memoize_cache.isKey( { [ p, q ] } )
                    value = memoize_cache( { [ p, q ] } );
                    value = value{ 1 };
                    return;
                end
                if p == 1
                    value = ( KP( :, :, q ) - KM( :, :, q ) ) ./ ( y * 1j );
                    value = real( value );
                elseif p == 3
                    value = - ( KP( :, :, q - 1 ) + KM( :, :, q - 1 ) ) + ( KP( :, :, q - 2 ) - KM( :, :, q - 2 ) ) ./ ( y * 1j );
                    value = value * 2 ./ ( y .^ 2 );
                    value = real( value );
                else
                    value = - ( p - 3 ) * T( p - 4, q - 2 ) + ( p - 2 ) * T( p - 2, q - 2 );
                    value = value ./ y .* ( p - 1 ) ./ y;
                    value = real( value );
                end
                memoize_cache( { [ p, q ] } ) = { value };
            end

            % T = memoize( @T_ );

            value = T( d - 2, d - 1 );
            value = value / ( ( 2 * pi ) ^ d ) * 2 * ( pi ^ ( ( d - 1 ) / 2 ) ) / gamma( ( d - 1 ) / 2 );
        end

        function [ value ] = get_value_positive_D_o_zero_y( self, t, d )
            % y = 0 special case
            arguments
                self ( 1, 1 ) C_FFPEHalfAlpha
                t    ( 1, 1 ) double          { mustBePositive }
                d    ( 1, 1 ) double          { mustBePositive, mustBeInteger }
            end
            component_base = 2 * sqrt( self.D_o * t );
            component = self.D_f * t / component_base;

            % Small d scenarios
            if d == 1
                value = F_erfcx( component ) / ( sqrt( 4 * pi * self.D_o * t ) );
                return;
            end

            K_recursion_base = 2 * self.D_o * t;
            K_recursion = - self.D_f / ( 2 * self.D_o );

            K = vpa( zeros( 1, d ) );
            K( 1, 1 ) = sqrt( vpa( pi ) ) / ( 2 * sqrt( self.D_o * t ) ) * F_erfcx( component ); % K_{0}
            K( 1, 2 ) = 1 / K_recursion_base + K_recursion * K( 1, 1 ); % K_{1}

            % if d == 2 % can be deleted
            %     value = K( 1, 2 ) / ( 2 * pi );
            %     return;
            % end

            for q = 3 : d
                K( 1, q ) = K_recursion * K( 1, q - 1 ) + ( q - 2 ) / K_recursion_base * K( 1, q - 2 );
            end

            value = K( 1, d ) / ( ( 2 * pi ) ^ d ) * 2 * ( pi ^ ( d / 2 ) ) / gamma( d / 2 );
        end

        function [ value ] = get_value_zero_D_o( self, y, t, d )
            arguments
                self ( 1, 1 ) C_FFPEHalfAlpha
                y    ( :, : ) double          { mustBeNonnegative }
                t    ( 1, 1 ) double          { mustBePositive }
                d    ( 1, 1 ) double          { mustBePositive, mustBeInteger }
            end
            tau = vpa( self.D_f .* t );
            y = vpa( y );
            k = vpa( ( d + 1 ) / 2 );
            p = vpa( pi );
            value = gamma( k ) ./ ( p .^ k ) .* tau ./ ( ( tau .^ 2 + y .^ 2 ) .^ k );
        end
    end
end

function [] = mustBeOdd( value )
    if mod( value, 2 ) == 0
        error_id = 'Value:notOdd';
        message = 'Value must be odd.';
        throwAsCaller( MException( error_id, message ) );
    end
end
