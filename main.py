import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QProgressBar
from PyQt5.uic import loadUi
from funciones import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from datetime import datetime


noticias = []
despoblacion = []
desp = []
no_despoblacion = []

clases = []
processed_text_entrenamiento = []
processed_text_testeo = []

nuevas = []
modelopath = []


cv = TfidfVectorizer()

#inicializar UI
class initial(QDialog):
    def __init__(self):
        super(initial,self).__init__()
        loadUi("main.ui", self)
        #definir elementos de la UI y acciones/funciones asociadas

        #Elementos Tab Entrenamiento
        #===========================

        #Btn Insertar primeros datos entrenamiento
        self.trainingAddFilesBtn.clicked.connect(self.insertarNoticiasEntrenamientoDespoblacion)

        #Btn Insertar segundos datos entrenamiento
        self.trainingAddFilesBtn2.clicked.connect(self.insertarNoticiasEntrenamientoNoDespoblacion)

        #Btn Preprocesamiento de texto
        self.procesarTextoBtn.clicked.connect(self.procesarTextoEntrenamiento)


        #ComboBox selector de Modelo a entrenar
        #self.chooseModelComboBox.activated.connect(self.elegirModeloEntrenamiento)

        #añadir elementos al comboBox y sus valores asociados
        self.chooseModelComboBox.addItem("KNN",1)
        self.chooseModelComboBox.addItem("Naive Bayes",2)
        self.chooseModelComboBox.addItem("Decision Tree",3)

        #Btn para entrenar el modelo seleccionado
        self.trainModelBtn.clicked.connect(self.entrenarModelo)

        #Elementos Tab Testeo
        #====================

        #Btn Insertar Datos Testeo
        self.testingFilesBtn.clicked.connect(self.insertarNoticiasTesteo)

        #Btn Seleccionar Modelo
        self.selectTestModelBtn.clicked.connect(self.elegirModeloTesteo)

        #Btn Ejecutar Testeo
        self.testBtn.clicked.connect(self.ejecutarTesteo)

    #funciones
    #=========

    # abrir dialog window para seleccionar los datos de entrenamiento
    def insertarNoticiasEntrenamientoDespoblacion(self):
        desp.append(self.openDialogBox())
        self.stepTipsField.setPlainText("Seleccionamos los directorios donde tenemos los archivos de texto que utilizaremos para entrenar nuestro modelo.")
        
        #cambiar self.procesarTextoBtn a habilitado
        self.trainingAddFilesBtn2.setEnabled(True)

    def insertarNoticiasEntrenamientoNoDespoblacion(self):
        no_despoblacion.append(self.openDialogBox())
        self.stepTipsField.setPlainText("Seleccionamos los directorios donde tenemos los archivos de texto que utilizaremos para entrenar nuestro modelo.")
        

        #cambiar self.procesarTextoBtn a habilitado
        self.procesarTextoBtn.setEnabled(True)

    def procesarTextoEntrenamiento(self):
        ingresar_noticias(desp[0], noticias)
        ingresar_noticias(no_despoblacion[0], noticias)


        ingresar_noticias(desp[0], despoblacion)

        #cambiar texto en self.stepTipsField
        self.stepTipsField.setPlainText("El preprocesamiento a realizar consta de 4 etapas:\n1. Tokenizar: separar las palabras que componen un texto, obteniendo como resultado una secuencia de tokens.\n2. Normalización: se pasa a minúsculas tdoos los tokens.\n3.Filtrado de stopwords: en esta etapa eliminamos  aquellas palabras con poco valor semántico, denominadas stopwords.\n4.Stemming: extraemos el lexema de los tokens restantes  (un ejemplo sería ‘cas-’ para la palabra ‘casero’)")

        #Procesamiento de texto
        texto_procesado(processed_text_entrenamiento, noticias)

        #Creación de arreglo de clases
        crea_clases(clases, processed_text_entrenamiento, despoblacion)
        #print(clases[len(despoblacion)+10])

        #cambiar self.trainModelBtn a habilitado
        self.trainModelBtn.setEnabled(True)

        self.stepTipsField.setPlainText("El preprocesamiento ha acabado")

    def openDialogBox(self):
        filenames = QFileDialog.getOpenFileNames()
        return filenames[0]

    def ejecutarTesteo(self):
        cargar_modelo(modelopath[0])


    #def elegirModeloEntrenamiento(self,index):
        #tomar valor actual del comboBox
     #   modelSelect = self.chooseModelComboBox.itemData(index)


    def insertarNoticiasTesteo(self):
        filepaths = self.openDialogBox()
        nuevas = []
        #ingresar noticias
        ingresar_noticias(filepaths, nuevas)

        #Procesamiento de texto
        texto_procesado(processed_text_testeo, nuevas)


    def elegirModeloTesteo(self):
        modelopath = []
        modelopath = self.openDialogBox()
        self.testBtn.setEnabled(True)

        




    def entrenamientoNaiveBayes(self):
        # Proceso TFIDF
        X_traincv = cv.fit_transform(processed_text_entrenamiento)
        # Partición de datos
        X_train, X_test, Y_train, Y_test = train_test_split(X_traincv, clases, test_size=0.15, random_state=324)

        #Modelos
        naive = naive_bayes(X_train, Y_train)
        print("####################### Test Score ##############################\n")
        test_score(naive, X_train, Y_train)

        # Creamos los datos a testear
        Y_true_naive, Y_pred_naive = datos_test(Y_test, naive, X_test)

        # Datos de los modelos
        print("###################### Accuracy ###############################\n")
        accuracy(Y_true_naive, Y_pred_naive)

        print("\n###################### Matriz de confusion ###############################\n")
        matrizconf(Y_true_naive, Y_pred_naive)

        #Guardamos modelo
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("dia:%d-%m-%Y,hora:%H-%M-%S")
        print("date and time =", dt_string)    
        guardar_modelo('modelos/naive_' + dt_string, naive)

    def entrenamientoArbolDecision(self):
        # Proceso TFIDF
        X_traincv = cv.fit_transform(processed_text_entrenamiento)
        #Partición de datos
        X_train, X_test, Y_train, Y_test = train_test_split(X_traincv, clases, test_size=0.15, random_state=324)

        #Modelos
        tree = decision_tree(X_train, Y_train)
        print("####################### Test Score ##############################\n")
        test_score(tree, X_train, Y_train)

        #Creamos los datos a testear
        Y_true_tree, Y_pred_tree = datos_test(Y_test, tree, X_test)


        #Datos de los modelos
        print("###################### Accuracy ###############################\n")
        accuracy(Y_true_tree, Y_pred_tree)

        #Matriz confusion
        print("\n###################### Matriz de confusion ###############################\n")
        matrizconf(Y_true_tree, Y_pred_tree)


        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("dia:%d-%m-%Y,hora:%H-%M-%S")
        print("date and time =", dt_string)    
        guardar_modelo('modelos/tree_' + dt_string, tree)


    def entrenamientoKnn(self):
        # Proceso TFIDF
        X_traincv = cv.fit_transform(processed_text_entrenamiento)
        #Partición de datos
        X_train, X_test, Y_train, Y_test = train_test_split(X_traincv, clases, test_size=0.15, random_state=324)

        #Modelos
        modknn = knn(X_train, Y_train)

        print("####################### Test Score ##############################\n")
        test_score(modknn, X_train, Y_train)

        #Creamos los datos a testear
        Y_true_knn, Y_pred_knn = datos_test(Y_test, modknn, X_test)


        #Datos de los modelos
        print("###################### Accuracy ###############################\n")

        accuracy(Y_true_knn, Y_pred_knn)

        #Matriz confusion
        print("\n###################### Matriz de confusion ###############################\n")
        matrizconf(Y_true_knn, Y_pred_knn)

        #Guardamos modelo
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("dia:%d-%m-%Y,hora:%H-%M-%S")
        print("date and time =", dt_string)    
        guardar_modelo('modelos/knn_' + dt_string, modknn)


    def entrenarModelo(self):
        #cambiar texto en self.stepTipsField
        self.stepTipsField.setPlainText(" Entrenando el modelo seleccionado")

        #tomar valor actual del comboBox
        modelSelect = self.chooseModelComboBox.currentData()
        print( "opcion seleccionada:",modelSelect)
        #no existe switch en python (o.o)
        if modelSelect == 1:
            self.entrenamientoKnn()

        if modelSelect == 2:
            self.entrenamientoNaiveBayes()

        if modelSelect == 3:
            self.entrenamientoArbolDecision()




#inicializar app
app=QtWidgets.QApplication(sys.argv)
#crear instancia clase initial
mainwindow=initial()


#Stack
widget=QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.show()
app.exec()
