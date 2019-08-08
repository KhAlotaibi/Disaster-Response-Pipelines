# Disaster Response Pipeline Project

### Summary
This project is part of Udacity's Data Science Nanodegree, the project includes the ETL process and the usage of Machine Learning Pipeline, and a web app to show the classification results. 

## File Description
#### 1- run.py
A flask web app to show classification results and visualization.
#### 2- process_data.py 
A python file contains the Extract,Transform,Load process 
#### 3- train_classifier.py
A python file where I load the data from the database and build the model and the pipeline. 



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
