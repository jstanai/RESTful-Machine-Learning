import os
import json

def get_config():
    try:
        with open(os.getenv("CONFIG_FILE")) as f:
            conf = json.load(f)
    except:
        print('You have not specified your environment variable \"CONFIG_FILE\"')
        # TODO: bring in default config
        conf = None

    return conf


    

