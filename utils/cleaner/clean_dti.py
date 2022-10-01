import pickle 


def read_cleaned_data(data_path):
    """
    Read the data files
    """

    with open(data_path, 'rb') as file:
        (RET,NET,FACTOR,YEAR_START,YEAR_END, G) = pickle.load(file)

    return (RET,NET,FACTOR,YEAR_START,YEAR_END, G)


def pp(a):
    print(a)