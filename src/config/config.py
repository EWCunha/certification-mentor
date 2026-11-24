from json import load, dump
from logging import getLogger
from os import remove, replace
from os.path import dirname
from tempfile import NamedTemporaryFile
from config.constants import CONFIG_PATH


class ConfigManager:
    """Configuration class for the application."""

    conf: dict

    def __init__(self, config_path: str = CONFIG_PATH):
        logger = getLogger(__name__)
        logger.info("Loading configuration")
        self.config_path = config_path

        with open(config_path, "r") as f:
            self.conf = load(f)

    def get(self, key: str):
        return self.conf.get(key, {})

    def save(self):
        directory = dirname(self.config_path) or "."
        with NamedTemporaryFile(
            "w", dir=directory, delete=False, suffix=".tmp"
        ) as tmp:
            tmp_path = tmp.name
            try:
                dump(self.conf, tmp, indent=4)
                tmp.write("\n")
            except Exception:
                tmp.close()
                remove(tmp_path)
                raise

        replace(tmp_path, self.config_path)
        return self


CONFIG = ConfigManager()
