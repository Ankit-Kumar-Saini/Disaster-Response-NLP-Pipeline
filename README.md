# Disaster Response NLP Pipeline 

### Table of Contents
1. [Dependencies](#dependency)
2. [Project Introduction](#introduction)
3. [Instructions for running the scripts](#instructions)
4. [Project Structure](#structure)
5. [File Descriptions](#files)
6. [Results](#results)
7. [Licensing, Authors, and Acknowledgements](#licensing)

## List of Dependencies<a name="dependency"></a>
The code should run with no issues using Python versions 3.
Other libraries used in this project are:
- numpy
- pandas
- flask
- nltk
- pickle
- matplotlib
- scikit-learn
- sqlalchemy

## Project Introduction<a name="introduction"></a>
The task of this project is to analyze disaster messages from Figure Eight and build a Machine Learning model that classifies disaster messages. The data set contains real messages that were sent during disaster events. A machine learning pipeline is created to categorize these events so that one can send the messages to an appropriate disaster relief agency. The project also includes a web app where an emergency worker can input a new message and get classification results in several categories.

## Instructions for running the scripts<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.
s
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Structure <a name="structure"></a>
1. app folder
	1. templates folder
		1. go.html
		2. master.html
	2. run.py

2. data folder
	1. disaster_categories.csv
	2. disaster_messages.csv
	3. DisasterResponse.db
	4. process_data.py

3. models folder
	1. classifier.pkl
	2. train_classifier.py

4. jupyter notebooks folder
	1. categories.csv
	2. messages.csv
	3. ETL Pipeline Preparation.ipynb
	4. ML Pipeline Preparation.ipynb

5. sample images folder

6. README.md

## File Descriptions <a name="files"></a>
1. The `app folder` contains files necessary for the functioning of the Web app. The templates folder contains two html files (`go.html` is used to render the information about the training data in the form of bar graphs and the classification results into 36 different categories while `master.html` is used to render the Web page). The `run.py` file runs the flask Web app. 

2. The `data folder` contains two csv files (disaster_messages.csv contains disaster messages and disaster_categories.csv contains 36 different categories into which disaster messages can be classified) and a sql database file `DisasterResponse.db` that contains the cleaned and processed disaster messages for training the classification model. The `process_data.py` script merges the two csv files into a single file, cleans the disaster messages and stores the cleaned/processed messages into a sql database.

3. The `train_classifier.py` script inside the `models folder` contains the code to load the cleaned disaster messages from the sql database, creates new features (number of words in each message, number of characters in each message, number of non stopwords in each message), builds Machine Learning Pipeline, performs GridSearchCV to find the best hyperparameter for the classification model, evaluates the trained model on test set and then saves the trained model as a pickle file to deploy on the Web app. The `classifier.pkl` file contains the trained model as pickle file.

4. The `jupyter notebooks` folder contains two jupyter notebooks. `ETL Pipeline Preparation.ipynb` notebook performs Extract, Load and Transform task on the messages and categories csv files after merging these two files. The `process_data.py` script is prepared using ETL notebook. `ML Pipeline Preparation.ipynb` contains Machine Learning Pipeline to classify disaster messages into 36 different categories. The `train_classifier.py` script is prepared using this notebook. 

5. The `sample_images` folder contains the images of visualizations from the ETL notebook and the working Web app for the purpose of quick demonstration in the results section below.

## Results<a name="results"></a>
`Some visualizations from this project`

- Number of messages in each genre
![alt text](https://github.com/Ankit-Kumar-Saini/Disaster-Response-NLP-Pipeline/blob/main/sample%20images/message_genre.PNG) 

- Number of messages in each category
![alt text](https://github.com/Ankit-Kumar-Saini/Disaster-Response-NLP-Pipeline/blob/main/sample%20images/categories.PNG) 

- Web app interface
![alt text](https://github.com/Ankit-Kumar-Saini/Disaster-Response-NLP-Pipeline/blob/main/sample%20images/web%20app%20interface.PNG) 

- Directing message to web app for classification
![alt text](https://github.com/Ankit-Kumar-Saini/Disaster-Response-NLP-Pipeline/blob/main/sample%20images/message.PNG) 

- Classification result of above message
![alt text](https://github.com/Ankit-Kumar-Saini/Disaster-Response-NLP-Pipeline/blob/main/sample%20images/message_classification.PNG) 

- Statistics of word and character counts of messages in the training data
![alt text](https://github.com/Ankit-Kumar-Saini/Disaster-Response-NLP-Pipeline/blob/main/sample%20images/stats.PNG) 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to Udacity for the data and python 3 notebook.




