import sys 
import os

import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

# import function from train_classifier.py
sys.path.append(os.path.abspath("/home/workspace/models"))
from train_classifier import *


app = Flask(__name__)

def tokenize(text):
    """
    This function will transform the raw text into clean text.
    
    It will tokenize input text message, lowercase each character
    and then apply lemmatization on lowercased tokens.

    Parameters:
    text (string): raw text message

    Returns:
    clean_tokens (list): clean tokenized text
    """

    # tokenize the input text
    tokens = word_tokenize(text)
    # create lemmatizer object
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    # iterate over the tokens
    for tok in tokens:
        # lemmatize, lowercase and strip spaces
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # number of messages in each genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # number of messages in each category
    category_counts = df.astype(bool).sum(axis = 0).iloc[2:]
    category_names = list(df.astype(bool).sum(axis = 0).iloc[2:].index)
    
    # number of words in each message
    num_words = df['message'].apply(lambda x: len(x.split()))
    message_ids = [i for i in range(len(df))]
    
    # number of characters in each message
    num_char = df['message'].str.len()
    
    # create visuals from the training data
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x = category_names,
                    y = category_counts
                )
            ],

            'layout': {
                'title': 'Number of messages in each category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x = message_ids,
                    y = num_words
                )
            ],

            'layout': {
                'title': 'Number of words in each message',
                'yaxis': {
                    'title': "Word Counts"
                },
                'xaxis': {
                    'title': "Message ID"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x = message_ids,
                    y = num_char
                )
            ],

            'layout': {
                'title': 'Number of characters in each message',
                'yaxis': {
                    'title': "Character Counts"
                },
                'xaxis': {
                    'title': "Message ID"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)



# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    
    # create dataframe
    query_df = pd.DataFrame([query], columns = ['message'])
    
    # create new features
    query_df = add_features(query_df)

    # use model to predict classification for query
    classification_labels = model.predict(query_df)[0]
    classification_results = dict(zip(df.columns[2:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()