import os
import locale
import commentjson as json

real_path = os.path.split(os.path.realpath(__file__))[0]


class Localizer:
    def __init__(self,language):
        self.language: str = language
        config_path = os.path.join(real_path, "..", "data", "config","language",language+"_language_config.json")

        if self.language == "auto":
            self.language = locale.getdefaultlocale()[0] # get the language code of the system (ex. zh_CN)
        self.language_map = {}


        self.file_is_exists = os.path.isfile(config_path)
        if self.file_is_exists:
            with open(config_path, "r", encoding="utf-8") as f:
                self.language_map.update(json.load(f))

    def __call__(self, key):
        if self.file_is_exists and key in self.language_map:
            return self.language_map[key]
        else:
            return key
