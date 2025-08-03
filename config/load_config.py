from os.path import join, basename, dirname, exists
import yaml
from config.get_project_root import get_project_root

config_yaml = join(dirname(__file__), "config.yaml")
assert exists(config_yaml), "Configure file does not exist"


def load_config():
    with open(config_yaml) as f:
        config = yaml.safe_load(f)
    root = get_project_root()
    return config, root


if __name__ == "__main__":
    config, project_root = load_config()
    print(project_root)
    print(config['PoseCorr']['train'])
    print(config['base_param'])
    print(config['base_param']['num_parts'])
