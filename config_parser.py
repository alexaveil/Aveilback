import configparser

def get_config_dict(path_config="config.ini"):
    config = configparser.ConfigParser()
    config.read(path_config)
    return config
