classdef C_TestWindowingFunction03 < C_WindowingFunction
    properties
        beta (1, 1) double {mustBePositive} = 1.5 % power parameter
    end

    methods
        function [ self ] = C_TestWindowingFunction03( M, gamma, beta )
            self@C_WindowingFunction( M, gamma );
            self.beta = beta;
        end
    end

    methods
        function [ result ] = get_value( self, x )
            s = ( abs( x ) - self.gamma * self.M ) / ( self.M * ( 1 - self.gamma ) );
            result = ( s <= 0 ) .* 1 + ( s > 0 & s < 1 ) .* exp( - 2 * exp( - 1 ./ ( abs( s ) .^ self.beta ) ) ./ ( abs( 1 - s ) .^ self.beta ) );
        end
    end
end
