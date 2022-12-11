import pandas as pd


def get_train_test(data, ycol="Outcome"):
    pass    
    
def load_diabetes_data(filename=None):
    if filename is None:
        filename = "data/diabetes.csv"
        
    df = pd.read_csv(filename)

    return df
