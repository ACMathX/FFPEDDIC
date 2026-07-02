classdef ( Abstract ) C_WindowingFunction < handle
    properties
        M     ( 1, 1 ) double { mustBePositive } = 10  % w(d) = 0 for |d| > M
        gamma ( 1, 1 ) double { mustBePositive } = 0.5 % w(d) = 1 for |d| < gamma M, 0 < gamma < 1
    end
    % all the derivatives of w(d) vanish at |d| = M and |d| = gamma M
    % w(d) exhibits a slow-rise from 0 to 1 as |d| goes from |d| = M to |d| = gamma M
    
    methods
        function [ self ] = C_WindowingFunction( M, gamma )
            if nargin < 1
                M = self.M;
            end
            if nargin < 2
                gamma = self.gamma;
            end
            self.validate_parameters( M, gamma );
            self.M = M;
            self.gamma = gamma;
        end
    end

    methods ( Abstract )
        [ result ] = get_value( self, x )
    end

    methods
        function [ ] = draw( self, number_of_point )
            if nargin < 2
                number_of_point = 100;
            end
            transition_length = self.M * ( 1 - self.gamma );
            extension_control = 2;
            x = linspace( self.gamma * self.M - transition_length / extension_control, self.M + transition_length / extension_control, number_of_point );
            w = self.get_value( x );
            plot( x, w );
            hold on;
            title( sprintf( 'M = %f, gamma = %f', self.M, self.gamma ) );
            ylim( [ - 0.05, 1.05 ] );
            line( [ self.gamma * self.M, self.gamma * self.M ], [ 0, 1 ], 'Color', 'red' );
            line( [ self.M, self.M ], [ 0, 1 ], 'Color', 'red' );
        end
    end

    methods ( Static, Access = private )
        function [ ] = validate_parameters( M, gamma )
            if gamma >= 1
                error( 'C_WindowingFunction:InvalidGamma', 'gamma must satisfy 0 < gamma < 1.' );
            end
            if M <= 0
                error( 'C_WindowingFunction:InvalidM', 'M must be positive.' );
            end
        end
    end
end
