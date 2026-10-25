from json import load
from logging import getLogger
from config.constants import CONFIG_PATH


class ConfigManager:
    """Configuration class for the application."""

    conf: dict

    def __init__(self, config_path: str = CONFIG_PATH):
        logger = getLogger(__name__)
        logger.info("Loading configuration")

        with open(config_path, "r") as f:
            self.conf = load(f)

    def get(self, key: str):
        return self.conf.get(key, {})


CONFIG = ConfigManager()
