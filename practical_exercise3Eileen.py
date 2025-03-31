import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('vader_lexicon')
import pandas as pd
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from nltk import word_tokenize, pos_tag, FreqDist, trigrams
from prettytable import PrettyTable
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
le = preprocessing.LabelEncoder()

DEBUG = False   

df = pd.read_csv('SMSSpamCollection.csv', encoding="latin-1")
df.columns = ['label', 'message']
df['message'] = df['message'].fillna('')

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)   
pd.set_option('display.width', 2000)   

def getFeatures(message):
    '''
    '''
    #length of message
    messagesLen = len(message)

    
    wordList = message.split()
    wordLen = len(wordList)
    
    prize_count = message.lower().split().count('prize')
    win_count = message.lower().split().count('win')
    text = re.sub("[^a-zA-Z ]", ' ', message)
    text = nltk.word_tokenize(text)
    tags = nltk.pos_tag(text)    
    
    nouns = 0
    verbs = 0
    adjectives = 0
    for eachTag in tags:
        if eachTag[1] in ["NN", "NNP", "NNPS"]:   
            nouns+=1
        elif eachTag[1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
            verbs+=1    
        elif eachTag[1] in ['JJ','JJR', 'JJS']:
            adjectives +=1
    try:
        ratio = nouns/verbs
    except:
        ratio = 0
    blob = TextBlob(message)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity  
    vader_analyzer = SentimentIntensityAnalyzer()
    vader_scores = vader_analyzer.polarity_scores(message)
    vader_polarity = vader_scores['compound']
    exclamations = message.count('!')
    question_marks = message.count('?')
    
    ''' Return the features as a list'''
    return [prize_count,  win_count, messagesLen, wordLen, nouns, verbs, adjectives, ratio, polarity, subjectivity, vader_polarity, exclamations, question_marks,]

def main():
    
    print("Create a Training and Testing Dataframes")
    dfTrain, dfTest = train_test_split(df, test_size=0.3, random_state=25)
    
    print("\nTraining DataFrame\n", dfTrain.head(5))
    print("\nTesting  DataFrame\n", dfTest.head(5))
    
    featureList = []  
    labelList = []   
    '''
    Process each row in the training dataframe
    '''
    print("\nProcessing Training Dataframe ...")
    
    for row in dfTrain.itertuples():
        rowLabel = row.label
        theMessage = row.message
        if rowLabel not in ['ham', 'spam']:
            continue
        features = getFeatures(theMessage)
        featureList.append(features)
        labelList.append(rowLabel)

    encodedLabels = le.fit_transform(labelList)

    # Create a K Nearest Neighbor Classifier
    print("Creating Nearest Neighbor Classifer Model ...")
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(featureList, encodedLabels)

    incorrect = 0
    correct   = 0
    totalExamined    = 0
    tbl = PrettyTable(["Known", "Predicted"])

    # Now test the modeul using the dfTrain dataframe
    print("\nApplying the Model to the data ...")
    for row in dfTest.itertuples():    
        tstMessage = row.message
        known = row.label
        features = getFeatures(tstMessage)

        prediction = model.predict([features])
        predicted  = le.inverse_transform(prediction)[0]
        

        if predicted == known:
            correct   += 1
        else:
            incorrect += 1

        totalExamined += 1 
        tbl.add_row([known, predicted])
    tbl.align='l'

    print(tbl.get_string(sortby="Known"))
    
    print("\nSummary Results")
     
    print("Total Examined:        ", totalExamined)
    print("Correctly Identified:  ", correct)
    print("Incorrectly Identified ", incorrect)
    print("Overall Accuracy:      ", (correct / totalExamined)* 100.0)

if __name__ == '__main__':
    main()
    print("\nScript End")