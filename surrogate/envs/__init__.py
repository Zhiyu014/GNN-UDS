from envs.scenario import *

def get_env(name):
    try:
        return eval(name)
    except:
        raise AssertionError("Unknown environment %s"%str(name))