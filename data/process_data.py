import re
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function will load the datasets specified by the filepaths 
    in the arguments passed to the function.

    It will merge the two datasets into a single dataframe using id 
    column.
    
    Parameters:
    messages_filepath (string): file path of messages dataset
    categories_filepath (string): file path of categories/labels dataset
        
    Returns:
    df (pandas dataframe): merged dataframe
        
    """
    # load the datasets
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    
    # merge the two datasets on 'id' column
    df = pd.merge(messages_df, categories_df, on = 'id', how = 'inner')
    
    return df


def clean_data(df):
    """
    This function will clean the dataframe so that each row represents 
    single observation and each column represents single variable.
    
    It splits the content of the categories column into separate columns 
    so that each category becomes a column in the dataframe. A value of 1 
    in any category column marks the presence of that category while a 
    value of 0 means the absence of that category. It also drops the duplicate
    entries from the dataframe.

    Parameters: 
    df (pandas dataframe): raw dataframe (dirty data)
        
    Returns:
    df (pandas dataframe): cleaned dataframe
    """

    # split each category into a column
    categories_df = df.categories.str.split(';', n = -1, expand = True)
    
    # select the first row from the categories_df
    row = categories_df.iloc[0, :]
    
    # extract column names from the row above
    category_colnames = row.apply(lambda x: x[:-2])
    
    # provide column names to the categories_df
    categories_df.columns = category_colnames
    
    # iterate over the columns of categories_df
    for column in categories_df:
        # set each value to be the last character of the string
        categories_df[column] = categories_df[column].str[-1]

        # convert column from string to numeric
        categories_df[column] = categories_df[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_df], axis = 1)

    # set id as the index of the dataframe
    df.set_index('id', inplace = True)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    # drop original column from the dataframe
    df.drop('original', axis = 1, inplace = True)
    
    # drop the rows with null values
    df.dropna(inplace = True)
    
    # Replace value of 2 in related column with 1
    df.loc[df[df.related == 2].index, 'related'] = 1
   
    return df


def save_data(df, database_filename):
    """
    This function will store the cleaned dataframe into a sql database.
    
    Parameters:
    df (pandas dataframe): cleaned dataframe
    database_filename (string): name of the database file

    Returns:
    None
    """

    # create connection to sqlite database
    engine = create_engine('sqlite:///' + database_filename) 
    # store dataframe in sql database
    df.to_sql('messages', engine, index = False)


def main():
	"""
	This function will call other helper functions defined above to 
	perform the data preprocessing task.
	"""

	# Check the number of arguments passed to the function
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