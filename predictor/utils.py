import pickle


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def load_pickle(pickle_path, default=None):
    try:
        with open(pickle_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    except FileNotFoundError as e:
        if default:
            return default
        else:
            raise e


def save_pickle(obj, pickle_path):
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(obj, pickle_file)
