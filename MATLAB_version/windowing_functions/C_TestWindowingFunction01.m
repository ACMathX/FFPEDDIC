classdef C_TestWindowingFunction01 < C_WindowingFunction
    methods
        function [ self ] = C_TestWindowingFunction01( M, gamma )
            if nargin < 1
                M = 10;
            end
            if nargin < 2
                gamma = 0.5;
            end
            self@C_WindowingFunction( M, gamma );
        end

        function [ result ] = get_value( self, x )
            s = ( abs( x ) - self.gamma * self.M ) / ( self.M * ( 1 - self.gamma ) );
            result = ( s <= 0 ) .* 1 + ( s > 0 & s < 1 ) .* exp( - 2 * exp( - 1 ./ ( abs( s ) ) ) ./ ( abs( 1 - s ) ) );
        end
    end
end
