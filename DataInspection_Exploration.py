import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('speed_data_data.csv')

def DataExploration(df):

    #Data Shape
    print('<Data Shape>')
    print(df.shape)
    print('\n')

    # Simple Information of Data(Column name, Non-Null Count, DataType)
    print('<Data Info>')
    print(df.info())
    print('\n')

    # Central Tendency and Variance Check
    print('<Data Describe>')
    print(df.describe())
    print('\n')
    
    # Data Values
    print('<Data Values>')
    print(df.value_counts())
    print('\n')

    # Show histogram of each column
    df.hist(figsize=(10, 9))
    plt.tight_layout()
    plt.show()
