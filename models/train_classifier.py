import sys
import numpy as np
import pandas as pd
import nltk
import pickle
import joblib
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score ,precision_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """Load X,Y,Category from database_filepath
    Args
        -database_filepath
    Return 
        -X,Y,and 36 category
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('etlpipelinetable',engine)
    X = df['message']
    Y =  df[['related-1', 'request-0', 'offer-0', 'aid_related-0', 
        'medical_help-0', 'medical_products-0', 'search_and_rescue-0',
        'security-0', 'military-0', 'child_alone-0', 'water-0', 'food-0', 
        'shelter-0', 'clothing-0', 'money-0', 'missing_people-0', 'refugees-0',
        'death-0', 'other_aid-0', 'infrastructure_related-0', 'transport-0', 
        'buildings-0', 'electricity-0', 'tools-0', 'hospitals-0', 'shops-0', 
        'aid_centers-0', 'other_infrastructure-0', 'weather_related-0', 'floods-0', 
        'storm-0', 'fire-0', 'earthquake-0', 'cold-0', 'other_weather-0', 
        'direct_report-0']]
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """Tokenize text,clean text,and Lemmatize
        Args
            -text 
        Return
            -cleaned token
    
    """
    stop_words = set(stopwords.words('english')) 
    token = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_token = []
    for tok in token:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stop_words: 
            clean_token.append(clean_tok)
    return clean_token

def build_model():
    """It return the cv that can be used as a model
    Args
        -None
    Return 
        -cv

    """
    pipeline = Pipeline([
                    ('tfidf',TfidfVectorizer(tokenizer = tokenize)),
                    ('clf',MultiOutputClassifier(RandomForestClassifier(n_estimators=12) ) )
])  
    parameters = {
    'clf__estimator__n_estimators': [5, 10],
    'clf__estimator__min_samples_split': [10, 20]
    }
    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model and gives the classification report
    Args
        -model,X_test,Y_test,category_names
    Return 
        -None
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))
def save_model(model, model_filepath):
    """Save the model
        Args
            -model,model_filepath
        Return
            -None
    
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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