from sklearn.preprocessing import FunctionTransformer

def generate_transformers(module, config) -> dict:
    """Return function transformer for every function included to be able to include in sklearn pipline. 
    If name is specified in the config file, then the key of the transfromer is the name itself, else the key
    is the method's name.

    Returns:
        dict: Fucntion transformer's objects indexed by the name.
    """
    function_transformer = {}
    for method in dir(module):
        if method[0] != "_":
            try:
                if isinstance(config[method], list):
                    for configs in config[method]:
                        name = method
                        if "name" in configs.keys():
                            name = configs['name']
                            del configs['name']
                        function_transformer[name] = FunctionTransformer(
                           module.__getattribute__(method), kw_args=configs)
                else:
                    name = method
                    if "name" in config[method].keys():
                        name = config[method]['name']
                        del config[method]['name']
                    function_transformer[name] = FunctionTransformer(
                        module.__getattribute__(method), kw_args=config[method])
            except KeyError:
                function_transformer[method] = FunctionTransformer(
                    module.__getattribute__(method))

    return function_transformer