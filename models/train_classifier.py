import re
import sys
import nltk
import pickle

nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import (CountVectorizer, 
                                             TfidfTransformer, 
                                             TfidfVectorizer)
from sklearn.metrics import (confusion_matrix, 
                             f1_score, 
                             classification_report, 
                             precision_score, 
                             recall_score)



def add_features(df):
    """
    This function will clean the input text messages and create additional 
    features to improve the performace of the model. 

    It will remove punctuations, lower case all the text messages and 
    apply lemmatization on the tokenized text. It will create new features 
    such as the length of the message, number of words in each message, 
    number of non stopwords in each message and average word length of each 
    message. Set of stopwords is also modified by removing negation words.
    Finally stopwords and words with length <= 2 have been removed from 
    the processed text messages.
    
    Parameters: 
    df (pandas dataframe): dataframe containing text messages
        
    Returns:
    df (pandas dataframe) : dataframe with new added features
    """

    # create a set of stopwords
    StopWords = set(stopwords.words('english'))
    
    # lowering and removing punctuation
    df['processed_text'] = df['message'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
    
    # apply lemmatization
    df['processed_text'] = df['processed_text'].apply(
        lambda x: ' '.join([WordNetLemmatizer().lemmatize(token) for token in x.split()]))
    
    # get the length of the message
    df['length'] = df['processed_text'].apply(lambda x: len(x))
    
    # get the number of words in each message
    df['num_words'] = df['processed_text'].apply(lambda x: len(x.split()))
    
    # get the number of non stopwords in each message
    df['non_stopwords'] = df['processed_text'].apply(
        lambda x: len([t for t in x.split() if t not in StopWords]))
    
    # get the average word length
    df['avg_word_len'] = df['processed_text'].apply(
        lambda x: np.mean([len(t) for t in x.split() if t not in StopWords]) \
        if len([len(t) for t in x.split() if t not in StopWords]) > 0 else 0)
    
    # update stop words (didn't want to remove negation)
    StopWords = StopWords.difference(
        ["aren't", 'nor', 'not', 'no', "isn't", "couldn't", "hasn't", 
         "hadn't", "haven't", "didn't", "doesn't", "wouldn't", "can't"])
    
    # remove stop words from processed text message
    df['processed_text'] = df['processed_text'].apply(
        lambda x: ' '.join([token for token in x.split() if token not in StopWords]))
    
    # filter the words with length > 2
    df['processed_text'] = df['processed_text'].apply(
        lambda x: ' '.join([token for token in x.split() if len(token) > 2]))
    
    return df


def load_data(database_filepath):
    """
    This function will load the stored dataset from the sql database.
    
    Parameters:   
    database_filepath (string): path of the database file
        
    Returns:
    df (pandas dataframe): dataframe loaded from the database
    """

    # create sql engine
    engine = create_engine('sqlite:///' + database_filepath)
    
    # load the stored dataset
    df = pd.read_sql('messages', con = engine)
    
    # create a list of category names
    category_names = df.columns[2:].tolist()
    
    # create additional features
    df = add_features(df)
    
    # list of features 
    features = ['processed_text', 'genre', 'length', 'num_words', 'non_stopwords', 'avg_word_len']
    
    # return features, labels and category names
    return (df[features], df[category_names], category_names)


def tokenizer(text):
  """
    This function will transform the raw text into clean text.
    
    It will tokenize input text message, lowercase each character
    and then apply lemmatization on lowercased tokens.

    Parameters:
    text (string): raw text message

    Returns:
    clean_tokens (list): clean tokenized text
    """

    # remove punctuations from raw text
    text = re.sub(r'[^\w\s]', '', text) 
    
    # tokenize filtered text
    tokens = word_tokenize(text)
    
    # create lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    # iterate over the tokens
    for token in tokens:
        # lemmatize, lowercase and strip spaces
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


class TextColumnSelector(BaseEstimator, TransformerMixin):
    """
    This class will create Transformer to select a single column from 
    the dataframe to perform transformations in sklearn pipeline. 
    This class inherits from BaseEstimator and TransformerMixin classes
    It will select columns containing text data from the dataframe when 
    provided with the name of the column.
    """

    def __init__(self, key):
        self.key = key
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return X[self.key]
    
    
class NumColumnSelector(BaseEstimator, TransformerMixin):
    """
    This class will create Transformer to select a single column from 
    the dataframe to perform transformations in sklearn pipeline. 
    This class inherits from BaseEstimator and TransformerMixin classes
    It will select columns containing numeric data from the dataframe when 
    provided with the name of the column.
    """

    def __init__(self, key):
        self.key = key
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return X[[self.key]]
    
    
class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    """
    This class inherits from BaseEstimator and TransformerMixin classes.
    It will create custom label binarizer for one hot encoding string 
    categorical columns.
    """

    def __init__(self, sparse_output = False):
        self.sparse_output = sparse_output
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        label_encoder = LabelBinarizer(sparse_output = self.sparse_output)
        return label_encoder.fit_transform(X)
    
   

def build_model():
    """
    This function will create separate pipelines to process individual columns 
    of the dataset and build the final model using individual pipelines.

    To select individual columns from the dataframe, this function will use 
    TextColumnSelector and NumColumnSelector classes. These separate pipelines
    will be combined and their data will be processed in parallel using 
    feature unions. StandardScaler is used to standardize the numeric data
    and TfidfVectorizer is used to transform the text data. Random Forest 
    is used as the classifier in the pipeline. GridSearchCV is used for
    hyperparameter tuning.
    """
    
    # pipeline to process num_words column
    num_words = Pipeline([
        ('selector', NumColumnSelector(key = 'num_words')),
        ('scaler', StandardScaler())
    ])

    # pipeline to process number of non_stopwords column
    num_non_stopwords = Pipeline([
        ('selector', NumColumnSelector(key = 'non_stopwords')),
        ('scaler', StandardScaler())
    ])

    # pipeline to process avg_word_len column
    avg_word_length = Pipeline([
        ('selector', NumColumnSelector(key = 'avg_word_len')),
        ('scaler', StandardScaler())
    ])

    # pipeline to process processed_text column
    message_processing = Pipeline([
        ('selecor', TextColumnSelector(key = 'processed_text')),
        ('tfidf', TfidfVectorizer(stop_words = 'english'))
    ])

    # pipeline to process length column
    length = Pipeline([
        ('selector', NumColumnSelector(key = 'length')),
        ('scaler', StandardScaler())
    ])

    # pipeline to process genre column
    # uncomment the lines below if genre column is provided at inference
#     genre = Pipeline([
#         ('selector', TextColumnSelector(key = 'genre')),
#         ('scaler', CustomLabelBinarizer())
#     ])
    
    # process all the pipelines in parallel using feature union
    feature_union = FeatureUnion([
        ('num_words', num_words),
        ('num_non_stopwords', num_non_stopwords),
        ('avg_word_length', avg_word_length),
        ('message_processing', message_processing),
        ('length', length)
        # ('genre_ohe', genre)
    ])

    # create final pipeline using Random Forest classifier
    final_pipeline = Pipeline([
        ('feature_union', feature_union),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # use GridSearch to find best hyperparameters for the model
    # prepare dictionary of parameters
    parameters = {'feature_union__message_processing__tfidf__max_df': [0.75, 1],
                  'feature_union__message_processing__tfidf__ngram_range': [(1, 1), (1, 2)],
                  'feature_union__message_processing__tfidf__use_idf': [True, False],
                  'clf__estimator__n_estimators': [200, 300]
                  }
    
    # create grid search object 
    grid_cv = GridSearchCV(final_pipeline, parameters, cv = 3, n_jobs = -1)
   
    return grid_cv
    
    
def evaluate_model(model, X_test, Y_test, category_names): 
    """
    This function will evaluate the trained model by calculating f1 score,
    precision score, recall score and classification report on test data
    
    Parameters:
    model: trained multioutput classifier model
    X_test (pandas dataframe): test features to evaluate model
    Y_test (numpy array): ground truth labels of test data
    category_names (list of strings): name of the categories
    """

    # make predictions on validation data
    y_pred = model.predict(X_test)
    
    # create dataframe from predictions
    preds_df = pd.DataFrame(y_pred, columns = category_names)
    
    # Create dictionary to hold the evaluation scores
    report = {}
    for col in Y_test.columns:
        report[col] = []
        # calculate precision score
        report[col].append(precision_score(Y_test[col], preds_df[col]))
        # calculate recall score
        report[col].append(recall_score(Y_test[col], preds_df[col]))
        # calculate f1 score
        report[col].append(f1_score(Y_test[col], preds_df[col]))
    
    # create dataframe from report dictionary
    report = pd.DataFrame(report)
    
    # print classification report 
    for i in range(len(category_names)):
        print("Precision, Recall, F1 Score for {}".format(Y_test.columns[i]))
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))



def save_model(model, model_filepath):
    """
    This function will save the trained model using pickle library.

    Parameters:
    model: trained multioutput classifier model
    model_filepath (string): name of the file to save model
    """

    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    """
    This function will call other helper functions to build, train,
    evaluate and save the model.
    """

    # Check the number of arguments passed to the function
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()