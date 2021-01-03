# Disaster Response ETL, ML pipelines and Web application.

### Motivation
The purpose of the project is to build a model for an API that classifies disaster messages.
Using the web app an emergency worker can input a new message and get classification results in several categories so to have an idea what kind of help is needed: "water", "shelter", "food", etc.  

The web app also displays visualizations of the data.

## Web application screenshots

![file1](https://github.com/himanshumangal09/UdacityDisasterResponse/blob/master/file1.JPG)

![file2](https://github.com/himanshumangal09/UdacityDisasterResponse/blob/master/file2.JPG)

![file3](https://github.com/himanshumangal09/UdacityDisasterResponse/blob/master/file3.JPG)

### Install
This project requires Python 3.x and the following Python libraries installed:

1. NumPy
2. Pandas
3. Matplotlib
4. Json
5. Plotly
6. Nltk
7. Flask
8. Sklearn
9. Sqlalchemy
10. Sys
11. Re
12. Pickle

You will also need to have software installed to run and execute an iPython Notebook

### Files:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- disaster_model.pkl  # saved model

- notebooks
|- ETL Pipeline Preparation.ipynb # etl exploration
|- ML Pipeline Preparation.ipynb # ML exploration

- README.md
```


### Code and data

-  process_data.py: This code extracts data from both CSV files: messages.csv (containing message data) and categories.csv (classes of messages) and creates an SQLite database containing a merged and cleaned version of this data.
-  train_classifier.py: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
-  ETL Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of process_data.py. process_data.py  automates this notebook.
-  ML Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. In particular, it contains the analysis used to tune the ML model and determine which model to use. train_classifier.py automates the model fitting process contained in this notebook.
-  disaster_messages.csv, disaster_categories.csv contain sample messages (real messages that were sent during disaster events) and categories datasets in csv format.
-  templates folder: This folder contains all of the files necessary to run and render the web app.
-  custom_transformer.py contains custom functions that were used in ML Pipeline Preparation.ipynb so to find best way of model tuning.

### Run
In a terminal navigate to the top-level project directory udacity-disaster-response/ (that contains this README) and run commands in the following sequence:

-  python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db  
-  python train_classifier.py DisasterResponse.db classifier.pkl
-  python run.py

Run the web application
Go to http://0.0.0.0:3001/ (if facing problems try http://localhost:3001 in a browser)

In the web app you may input any text message (English) and it will categorize it among 35 classes.

### Data and cleaning details:
The first part of my data pipeline is the Extract, Transform, and Load process. Here, I read the dataset, clean the data, and then store it in a SQLite database. I have done the data cleaning with pandas. To load the data into an SQLite database, I have use the pandas dataframe .to_sql() method.


### Data observations
As can be seen from test data visualization most of the classes (categories) are highly imbalanced. This affects model F1 prediction score. One using this project should take this into consideration and apply measures like synthetic data generation, model selection and parameters fine-tuning, etc.     


