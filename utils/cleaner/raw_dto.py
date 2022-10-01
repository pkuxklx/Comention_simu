import pickle

def save_result(x, filename):
    """
    save result of x to the specific location
    """
    with open(filename, 'wb') as file:
        pickle.dump(x, file)