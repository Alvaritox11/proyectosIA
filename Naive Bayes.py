import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from nltk.corpus import stopwords
from sklearn.utils import shuffle

########################################################################
# Creacion vocabulario
########################################################################

def vocabulary_creation(df, percent=1.0):
    vocabulario = {}
    
    total_tweets = len(df['tweetText'])
    target_count = int(total_tweets * percent)
    
    for tweet in df['tweetText']:
        palabras = tweet.split()
        for palabra in palabras:
            if palabra in vocabulario:
                vocabulario[palabra] += 1
            else:
                vocabulario[palabra] = 1
    
    stop_words = stopwords.words('english')
    for k in stop_words:
        vocabulario.pop(k, None)
    
    subvocabulario = dict(list(vocabulario.items())[:target_count])
    return subvocabulario

########################################################################
# fit
########################################################################

def learn_naive_bayes_tweets(df, vocabulario):
    labels = df['sentimentLabel'].unique() # vj in V
    prob_condicionades = {label: defaultdict(float) for label in labels}
    probabilitat_label = {}
    for label in labels:
        df_aux = df[df['sentimentLabel'] == label] # docsj
        probabilitat_label[label] = df_aux.shape[0] / df.shape[0] # P(vj)
        texto_completo = ' '.join(df_aux['tweetText'].dropna().astype(str)) # Textj
        n = len(texto_completo.split()) 
        for palabra in vocabulario.keys():
            nk = texto_completo.split().count(palabra)
            prob_condicionades[label][palabra] = nk / n
            
    return prob_condicionades, probabilitat_label

def learn_naive_bayes_tweets_Smoothing(df, vocabulario):
    labels = df['sentimentLabel'].unique() # vj in V
    prob_condicionades = {label: defaultdict(float) for label in labels}
    probabilitat_label = {}
    for label in labels:
        df_aux = df[df['sentimentLabel'] == label] # docsj
        probabilitat_label[label] = df_aux.shape[0] / df.shape[0] # P(vj)
        texto_completo = ' '.join(df_aux['tweetText'].dropna().astype(str)) # Textj
        n = len(texto_completo.split()) 
        for palabra in vocabulario.keys():
            nk = texto_completo.split().count(palabra)
            prob_condicionades[label][palabra] = (nk + 1) / (n + len(vocabulario))
            
    return prob_condicionades, probabilitat_label

########################################################################
# predict
########################################################################

def classify_naive_bayes_tweets(tweet, vocabulario, probabilitats, probabilitat_label):
    paraules = tweet.split()
    positions = [paraula for paraula in paraules if paraula in vocabulario]
    prob_posteriors = {label: probabilitat_label[label] for label in probabilitat_label}
    
    for label in prob_posteriors:
        for paraula in positions:
            prob_posteriors[label] *= probabilitats[label].get(paraula, 1/(len(vocabulario) + sum(probabilitats[label].values())))
    
    v_nb = max(prob_posteriors, key=prob_posteriors.get)
    return v_nb

########################################################################
# cross validation
########################################################################

def cross_validation(df, train, percent):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    predictions = []
    y_tests = []
    accuracy_mean = []
    
    for i, (train_index, test_index) in enumerate(kf.split(train)):
        X_train, X_test = train.iloc[train_index]['tweetText'], train.iloc[test_index]['tweetText']
        y_train, y_test = train.iloc[train_index]['sentimentLabel'], train.iloc[test_index]['sentimentLabel']

        train_df = pd.DataFrame({'tweetText': X_train, 'sentimentLabel': y_train})

        vocabulario = vocabulary_creation(train_df, percent)
        probabilitats, probabilitat_label = learn_naive_bayes_tweets(train_df, vocabulario)
        
        predicted_categories = [classify_naive_bayes_tweets(tweet, vocabulario, probabilitats, probabilitat_label) for tweet in X_test]
        predictions.extend(predicted_categories)
        
        accuracy = accuracy_score(y_test, predicted_categories)
        accuracy_mean.append(accuracy)

        y_tests.extend(y_test)
        
        print(f'Accuracy para el fold {i + 1}: {accuracy}')

    return predictions, y_tests, accuracy_mean


########################################################################
# Ejercicio 1
########################################################################

def ejercicio1(df):
    print("INICI EXERCICI 1: ")
    print(" ")
    print("Train = 60% de les dades, Test = 40% de les dades")
    train, test = train_test_split(df, test_size=0.4, random_state=42)
    
    print("INICI CROSS VALIDATION:")
    predictions, y_tests, accuracy_mean = cross_validation(df, train, 1.0)
    print("Promig de la validació creuada:", sum(accuracy_mean) / len(accuracy_mean))
    print("FINAL CROSS VALIDATION")
    print(" ")

    print("INICI TEST: ")
    vocabulario = vocabulary_creation(train)
    probabilitats, probabilitat_label = learn_naive_bayes_tweets(train, vocabulario)
    predicted_test = [classify_naive_bayes_tweets(tweet, vocabulario, probabilitats, probabilitat_label) for tweet in test['tweetText']]
    y_test = test['sentimentLabel']
    
    print("RESULTATS FINALS:")
    report = classification_report(y_test, predicted_test)
    print(report)
    print(" ")
    print("FINAL EXERCICI 1")


########################################################################
# Ejercicio 2 Apartat 1
########################################################################

def ejercicio21(df, smoothing=False):
    print("INICI EXERCICI 2.1: ")
    print(" ")
    print("Train = 80% de les dades, Test = 20% de les dades")
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    print("INICI TEST: ")
    vocabulario = vocabulary_creation(train)
    if smoothing:
        probabilitats, probabilitat_label = learn_naive_bayes_tweets_Smoothing(train, vocabulario)
    else:
        probabilitats, probabilitat_label = learn_naive_bayes_tweets(train, vocabulario)
    predicted_test = [classify_naive_bayes_tweets(tweet, vocabulario, probabilitats, probabilitat_label) for tweet in test['tweetText']]
    y_test = test['sentimentLabel']
    
    print("RESULTATS FINALS:")
    report = classification_report(y_test, predicted_test)
    print(report)
    print(" ")
    print("FINAL EXERCICI 2.1")


########################################################################
# Ejercicio 2 Apartat 2
########################################################################

def ejercicio22(df, smoothing=False):
    print("INICI EXERCICI 2.2: ")
    print(" ")
    accuracies = []
    dictionary_sizes = []
    print("Fixem el tamany del train al 50% de les dades")
    train, test = train_test_split(df, test_size=0.5, random_state=42)
    print("Incrementarem el tamany del diccionari un 10% cada iteració")
    print(" ")
    for i in range(2, 11, 2):
        print(f"Tamany del diccionari: {i * 10}%")
        print(" ")
        dictionary_size = i * 0.1
        print("INICI TEST: ")
        vocabulario = vocabulary_creation(train, i * 0.1)
        if smoothing:
            probabilitats, probabilitat_label = learn_naive_bayes_tweets_Smoothing(train, vocabulario)
        else:
            probabilitats, probabilitat_label = learn_naive_bayes_tweets(train, vocabulario)
        predicted_test = [classify_naive_bayes_tweets(tweet, vocabulario, probabilitats, probabilitat_label) for tweet in test['tweetText']]
        y_test = test['sentimentLabel']
        
        print("RESULTATS FINALS:")
        #report = classification_report(y_test, predicted_test)
        #print(report)
        accuracy = accuracy_score(y_test, predicted_test)
        print(f"L'accuracy per aquest conjunt és de: {accuracy}")
        accuracies.append(accuracy)
        dictionary_sizes.append(dictionary_size * 100)
        print(" ")
    
    plt.plot(dictionary_sizes, accuracies, marker='o')
    plt.xlabel("Tamany del diccionari")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Tamany del diccionari")
    plt.show()
    
    print("FINAL EXERCICI 2.2")

########################################################################
# Ejercicio 2 Apartat 3
########################################################################

def ejercicio23(df, smoothing=False):
    print("INICI EXERCICI 2.3: ")
    print(" ")
    accuracies = []
    train_sizes = []
    print("Fixem el tamany del diccionari al 100% de les dades")
    train, test = train_test_split(df, test_size=0.8, random_state=42)
    print("Incrementarem el tamany del train un 10% cada iteració")
    print(" ")
    for i in range(2, 11, 2):
        print(f"Tamany del train: {i * 10}%")
        print(" ")
        train_size = i * 0.1
        print("INICI TEST: ")
        vocabulario = vocabulary_creation(train)
        if smoothing:
            probabilitats, probabilitat_label = learn_naive_bayes_tweets_Smoothing(train[:int(i * 0.1 * len(train))], vocabulario)
        else:
            probabilitats, probabilitat_label = learn_naive_bayes_tweets(train[:int(i * 0.1 * len(train))], vocabulario)
        predicted_test = [classify_naive_bayes_tweets(tweet, vocabulario, probabilitats, probabilitat_label) for tweet in test['tweetText']]
        y_test = test['sentimentLabel']
        
        print("RESULTATS FINALS:")
        accuracy = accuracy_score(y_test, predicted_test)
        print(f"L'accuracy per aquest conjunt és de: {accuracy}")
        accuracies.append(accuracy)
        train_sizes.append(train_size * 100)
        print(" ")
        
    plt.plot(train_sizes, accuracies, marker='o')
    plt.xlabel("Tamany del train")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Tamany del train")
    plt.show()
    
    print("FINAL EXERCICI 2.3")


########################################################################
# Ejercicio 2 Apartat 3
########################################################################

def ejercicio3(df):
    ejercicio21(df, True)
    print(" ")
    ejercicio22(df, True)
    print(" ")
    ejercicio23(df, True)

########################################################################
# main
########################################################################

if __name__ == '__main__':
    # Cargar datos
    df = pd.read_csv("FinalStemmedSentimentAnalysisDataset.csv", sep=';')
    df = shuffle(df, random_state=42) 
    # Elmininacion de nans
    df = df.dropna(subset=['tweetText'])
    df = df.reset_index(drop=True)

    print("Balanceig de les dades:")
    print(f"N mostres amb sentimentLabel igual a 0: {len(df[df['sentimentLabel'] == 0])}")
    print(f"N mostres amb sentimentLabel igual a 1: {len(df[df['sentimentLabel'] == 1])}")
    print(" ")
    print(" ")

    ejercicio1(df)
    print(" ")
    ejercicio21(df)
    print(" ")
    ejercicio22(df)
    print(" ")
    ejercicio23(df)
    print(" ")
    ejercicio3(df)


