#Librerías
import glob
import numpy as np
import pickle

#Librerías NLTK
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#Librerías SKLEARN
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_validate


#Ingresar txt a dataset
def ingresar_noticias(direccion, dataset):
    for ficheroIN in (glob.glob(direccion)):
        dataset.append(ficheroIN)

#Imprimir el documento
def string_doc(id, noticias):
    archivo = open(noticias[id], "r", encoding='utf-8', errors='ignore')
    texto = archivo.read().strip()
    return texto

##############################################################################
#Remover puntuación
def remove_punctuation ( text ):
    return re.sub('[%s]' % re.escape(string.punctuation), ' ', text)


#Pasar todo a minúsculas
def convert_lower_case(data):
    return np.char.lower(data)

#Remover lista de parada
def remove_stop_words(data):
    stop_words = stopwords.words('spanish')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

#Quitar puntuacion
def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

#Quitar apostrofe
def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

#Stem
def stemming(data):
    stemmer= SnowballStemmer('spanish')

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

#############################################################################

#Función Preprocesado de datos
def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = remove_stop_words(data)
    return data

#Texto procesado
def texto_procesado(noticias):
    processed_text = []
    for i in range(len(noticias)):
        archivo = open(noticias[i], "r", encoding='utf-8', errors='ignore')
        texto = archivo.read().strip()
        archivo.close()
        processed_text.append(preprocess(texto))

    print('Cantidad de texto procesado: ', len(processed_text))
    
    return processed_text


#Creación de clases

def crea_clases(clases, processed_text, despoblacion):
    for i in range(0, len(despoblacion)):
        clases.append("Despoblacion")  # Despoblacion

    for i in range(len(despoblacion), len(processed_text)):
        clases.append("No despoblacion")  # No despoblacion
    
    le = preprocessing.LabelEncoder()
    le.fit(clases)

    return clases

#Proceso TFIDF
def tfid(processed_text, cv):
    X_traincv = cv.fit_transform(processed_text)
    return X_traincv

def tfid_fit(processed_text, cv):
    X_traincv = cv.transform(processed_text)
    return X_traincv


#Naive Bayes
def naive_bayes(X_traincv, clases):
    mnb = MultinomialNB()
    mnb.fit(X_traincv, clases)
    return mnb

#Decision Tree Classifier
def decision_tree(X_traincv,  clases):
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_traincv, clases)
    return tree

#RandomForest
def ramdonforest(X_traincv, clases):
    random_forest = RandomForestClassifier(random_state=0)
    random_forest.fit(X_traincv, clases)
    return random_forest

#Test Score
def test_score(modelo, X_traincv,  clases):
    val_cruzada = cross_validate(modelo, X_traincv, clases, cv=3)
    print(val_cruzada['test_score'])

#Creación de datos a testear
def datos_test(Y_test, modelo, X_test):
    y_true_dt, y_pred_dt = Y_test, modelo.predict(X_test)
    return y_true_dt, y_pred_dt

#Matriz de confusión
def matrizconf(y_true_dt,y_pred_dt):
    print(confusion_matrix(y_true_dt, y_pred_dt))

#Accuracy
def accuracy(y_true_dt,y_pred_dt):
    print(accuracy_score(y_true_dt, y_pred_dt) * 100)

#Resultado de modelo
def prediccion(modelo, noticia):
    pred = modelo.predict(noticia)
    return pred

#Guardar modelo
def guardar_modelo(direccion, modelo):
    with open(direccion+'.pk1', 'wb') as f:
        pickle.dump(modelo, f)

def cargar_modelo(direccion):
    with open(direccion, 'rb') as f:
        modelo = pickle.load(f)
    return modelo



