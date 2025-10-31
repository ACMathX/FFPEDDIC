function [ value ] = F_erfcx( z )
    z = vpa( z );
    value = exp( z .^ 2 ) .* erfc( z );
end
