import torch


class WindowingFunction:
    def __init__( self, M, gamma = 0.5 ):
        self.M = float( M )
        self.gamma = float( gamma )
        self.validate_parameters()

    def validate_parameters( self ):
        if self.M <= 0:
            raise ValueError( 'M must be positive.' )
        if self.gamma <= 0 or self.gamma >= 1:
            raise ValueError( 'gamma must satisfy 0 < gamma < 1.' )

    def get_s( self, x ):
        return ( torch.abs( x ) - self.gamma * self.M ) / ( self.M * ( 1.0 - self.gamma ) )

    def get_value( self, x ):
        raise NotImplementedError


class TestWindowingFunction01( WindowingFunction ):
    def get_value( self, x ):
        s = self.get_s( x )
        result = torch.ones_like( s )

        outside_flag = s >= 1.0
        transition_flag = ( s > 0.0 ) & ( s < 1.0 )

        result[ outside_flag ] = 0.0
        if torch.any( transition_flag ):
            s_transition = s[ transition_flag ]
            result[ transition_flag ] = torch.exp(
                - 2.0
                * torch.exp( - 1.0 / torch.abs( s_transition ) )
                / torch.abs( 1.0 - s_transition )
            )
        return result


class TestWindowingFunction02( WindowingFunction ):
    def get_value( self, x ):
        s = self.get_s( x )
        result = torch.ones_like( s )

        outside_flag = s >= 1.0
        transition_flag = ( s > 0.0 ) & ( s < 1.0 )

        result[ outside_flag ] = 0.0
        if torch.any( transition_flag ):
            s_transition = s[ transition_flag ]
            result[ transition_flag ] = torch.exp(
                - 2.0
                * torch.exp( - 1.0 / ( torch.abs( s_transition ) ** 2 ) )
                / ( torch.abs( 1.0 - s_transition ) ** 2 )
            )
        return result


class TestWindowingFunction03( WindowingFunction ):
    def __init__( self, M, gamma = 0.5, beta = 1.5 ):
        self.beta = float( beta )
        super().__init__( M, gamma )

    def get_value( self, x ):
        s = self.get_s( x )
        result = torch.ones_like( s )

        outside_flag = s >= 1.0
        transition_flag = ( s > 0.0 ) & ( s < 1.0 )

        result[ outside_flag ] = 0.0
        if torch.any( transition_flag ):
            s_transition = s[ transition_flag ]
            result[ transition_flag ] = torch.exp(
                - 2.0
                * torch.exp( - 1.0 / ( torch.abs( s_transition ) ** self.beta ) )
                / ( torch.abs( 1.0 - s_transition ) ** self.beta )
            )
        return result
