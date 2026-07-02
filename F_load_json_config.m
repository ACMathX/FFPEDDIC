function [ config ] = F_load_json_config( file_name )
    if nargin < 1 || isempty( file_name )
        config = struct();
        return;
    end

    if ~ isfile( file_name )
        error( 'F_load_json_config:MissingConfig', 'Config file not found: %s', file_name );
    end

    config_text = fileread( file_name );
    config = jsondecode( config_text );
end
