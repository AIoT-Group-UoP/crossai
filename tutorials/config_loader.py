import yaml


def load_config(config_file_path: str):
    """
    Load configuration from a YAML file.

    Args:
        config_file_path: Path to the YAML configuration file.

    Returns:
     config: Configuration object if the file is read successfully,
        None otherwise.
    """
    try:
        with open(config_file_path, "r") as stream:
            config = yaml.safe_load(stream)
            return config
    except FileNotFoundError:
        print(f"Configuration file not found at {config_file_path}")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
    return None
