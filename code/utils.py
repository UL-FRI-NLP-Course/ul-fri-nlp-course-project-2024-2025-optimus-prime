import json
import os


def get_param(field):
    """
    Read parameters from the config.json file.
    """

    with open("config.json", "r") as file:
        config = json.load(file)

        return config[field]
