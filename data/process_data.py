import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    """load the data from command argument
        
        Parameters:
        csv file :messages_filepath
        csv file :categories_filepath
        
        Returns:
        Merged csv file df
        
        """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    
    return df 

def clean_data(df):
    """Clean the csv file df
        
        Parameters:
        csv file
        
        Returns:
        Cleand csv file df
        
        """
    
    categories = df.categories.str.split(pat=';',expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x:x[:-2])
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)
    
    df = df.drop('categories',axis=1)
    
    df = pd.concat([df,categories],axis=1)
    
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """Create an engine and load the df to sql
        
        Parameters:
        df
        the name of the database from a python argument
        
        Returns:
        No returns
        
        """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DataFrame', engine, index=False)  


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
