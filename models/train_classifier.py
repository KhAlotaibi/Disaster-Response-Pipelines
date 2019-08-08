import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report,confusion_matrix
import pickle
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath, table= 'DataFrame'):
    """load the data from database and the table name
        
        Parameters:
        The filepath of the database,
        The name of the table
        
        Returns:
        X,Y extraxted from the table, and category columns names
        
        
        """
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table, engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_columns = []
    for i in Y.columns:
        category_columns.append(i)
    return X,Y,category_columns
    


def tokenize(text):
    """Function to tokenize the text messages
        
        Parameters:
        text
        
        Returns:
        cleaned tokenized text as a list object
        
        
        """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Function to build the model, create the pipeline and implement gridsearchcv
        
        Parameters:
        No parameters
        
        Returns:
        Model
        
        
        """
    
    #Create the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        
    ])
    
    # Grid Search
    parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            'clf__estimator__min_samples_split': [2, 4],
            'tfidf__use_idf': (True, False)
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_columns):
    """Function to evaluate the model and return the classification report
        
        Parameters:
        The model,
        X_test,
        Y_test,
        category columns names
        
        
        Returns:
        Prints the classification report
        
        
        """
    
    y_pred = model.predict(X_test)
    
    for col in range(0, len(category_columns)):
        print(category_columns[col])
        print(classification_report(Y_test[category_columns[col]], y_pred[:, col]))


def save_model(model, model_filepath):
    """Function to save the model as a pickle file
        
        Parameters:
        The model,
        The model file path as python argument
        
        
        Returns:
        Save the model.
        
        
        """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_columns = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_columns)

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
