import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
from mainsofi import * 
from funciones import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

noticias = []
despoblacion = []
processed_text_entrenamiento = []
processed_text_testeo = []
nuevas = []
clases = []


#inicializar UI
class initial(QDialog):
    def __init__(self):
        super(initial,self).__init__()
        loadUi("main.ui", self)
        #definir elementos de la UI y acciones/funciones asociadas
        """
        #Tab entrenamiento
        #=================
        #btn add files entrenamiento
        self.trainingAddFilesBtn.clicked.ingresar_noticias()
        #btn comparar modelos
        self.comparModelsBtn.clicked# funcion comparar modelos seleccionados
        #btn entrenar 1er modelo
        self.trainingModel1Btn.clicked.knn() #poder cambiar de modelo lo mismo? con un comboBox?
        #btn entrenar 2o modelo
        self.trainingModel2Btn.clicked.decision_tree() #poder cambiar de modelo lo mismo? con un comboBox?
        #campo texto explicacion proceso
        self.stepTipsField #funcion cambiar texto al clickar distintos botones
        #frame donde mostrar resultados/comparativa de los modelos
        self.frameModels    #?

        #Tab testeo
        #=================
        #btn datos testeo
        self.testingFilesBtn.clicked.datos_test()
        #comboBox seleccionar modelo testeo
        self.testingComboBox    #?
        #btn ejecutar testeo
        self.testBtn.clicked #?
        #frame resultados testeo
        self.testFrame #?
        """

        self.trainingAddFilesBtn.clicked.connect(self.insertarNoticiasEntrenamiento)
        self.testingFilesBtn.clicked.connect(self.insertarNoticiasTesteo)

        self.trainingModelNB.clicked.connect(self.entrenamientoNaiveBayes)
        self.trainingModelAD.clicked.connect(self.entrenamientoArbolDecision)
        self.trainingModelKnn.clicked.connect(self.entrenamientoKnn)


    def insertarNoticiasEntrenamiento(self):
        #Noticias
        ingresar_noticias("despoblación/*.txt", noticias)
        ingresar_noticias("no_despoblación/*.txt", noticias)

        ingresar_noticias("despoblación/*.txt", despoblacion)


        #Procesamiento de texto
        processed_text_entrenamiento = texto_procesado(noticias)

        #Creación de arreglo de clases
        clases = crea_clases(processed_text_entrenamiento, despoblacion)

    def insertarNoticiasTesteo(self):
        #Noticias
        
        ingresar_noticias("unlabeled/*.txt", nuevas)

        #Procesamiento de texto
        processed_text_testeo = texto_procesado(nuevas)

    def entrenamientoNaiveBayes(self):
        cv = TfidfVectorizer()
        #TFIDF_noticias
        X_traincv = tfid(processed_text_entrenamiento, cv)
        #Partición de datos
        X_train, X_test, Y_train, Y_test = train_test_split(X_traincv, clases, test_size=0.15, random_state=324)

        #Modelos
        naive = naive_bayes(X_train, Y_train)

        #Creamos los datos a testear
        Y_true_naive, Y_pred_naive = datos_test(Y_test, naive, X_test)

        #Datos de los modelos
        accuracy(Y_true_naive,Y_pred_naive)

        #Matriz confusion
        matrizconf(Y_true_naive,Y_pred_naive)


    def entrenamientoArbolDecision(self):
        cv = TfidfVectorizer()
        #TFIDF_noticias
        X_traincv = tfid(processed_text_entrenamiento, cv)
        #Partición de datos
        X_train, X_test, Y_train, Y_test = train_test_split(X_traincv, clases, test_size=0.15, random_state=324)

        #Modelos
        tree = decision_tree(X_train, Y_train)

        #Creamos los datos a testear
        Y_true_tree, Y_pred_tree = datos_test(Y_test, tree, X_test)


        #Datos de los modelos
        accuracy(Y_true_tree, Y_pred_tree)

        #Matriz confusion
        matrizconf(Y_true_tree, Y_pred_tree)
    
    def entrenamientoKnn(self):
        cv = TfidfVectorizer()
        #TFIDF_noticias
        X_traincv = tfid(processed_text_entrenamiento, cv)
        #Partición de datos
        X_train, X_test, Y_train, Y_test = train_test_split(X_traincv, clases, test_size=0.15, random_state=324)

        #Modelos
        knn = knn(X_train, Y_train)

        #Creamos los datos a testear
        Y_true_knn, Y_pred_knn = datos_test(Y_test, knn, X_test)


        #Datos de los modelos
        accuracy(Y_true_knn, Y_pred_knn)

        #Matriz confusion
        matrizconf(Y_true_knn, Y_pred_knn)





#inicializar app
app=QtWidgets.QApplication(sys.argv)
#crear instancia clase initial
mainwindow=initial()


#Stack 
widget=QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.show()
app.exec()
