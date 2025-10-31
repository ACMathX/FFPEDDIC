classdef C_TestWindowingFunction01 < C_WindowingFunction
    methods
        function [ result ] = get_value( self, x )
            s = ( abs( x ) - self.gamma * self.M ) / ( self.M * ( 1 - self.gamma ) );
            result = ( s <= 0 ) .* 1 + ( s > 0 & s < 1 ) .* exp( - 2 * exp( - 1 ./ ( abs( s ) ) ) ./ ( abs( 1 - s ) ) );
        end
    end
end
