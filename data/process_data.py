import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
        Args
         -messages_filepath and categories_filepath
        Return 
         -return Merged data set
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')

def clean_data(df):
    """
    Args
        -dataframe that should be clean
    Return 
         - cleaned datframe
                -drop duplicate
                 - cleans up the categories column
    """
    col_names = str(df['categories'][0]).split(";")
    categories = pd.DataFrame(df['categories'].str.split(";",expand=True) )
    categories.columns = col_names
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str[-1]
    # convert column from string to numeric
        categories[column] = categories[column].astype('int32')
    categories.head()
    df.drop(columns=['categories'],axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    duplicate_rows = df[df.duplicated()]
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """
    It saves the dataframe in a filename
    Args
        -dataframe
        -database-filename
    Return
        -None
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('etlpipelinetable', engine, index=False,if_exists='replace')   
    return None
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')
if __name__ == '__main__':
    main()