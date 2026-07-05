import json


def load_config( file_name = None ):
    if file_name is None or file_name == '':
        return {}

    with open( file_name, 'r', encoding = 'utf-8' ) as config_file:
        return json.load( config_file )


def display_config( config ):
    print( json.dumps( config, indent = 4, sort_keys = True ) )
