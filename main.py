import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from funciones import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


noticias = []
despoblacion = []

clases = []
processed_text_entrenamiento = []
processed_text_testeo = []

nuevas = []

cv = TfidfVectorizer()

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
        #Btn Insertar datos entrenamiento
        self.trainingAddFilesBtn.clicked.connect(self.insertarNoticiasEntrenamiento)

        #Btn Preprocesamiento de texto
        self.procesarTextoBtn.clicked.connect(self.procesarTexto)

        #ComboBox selector de Modelo a entrenar
        #self.chooseModelComboBox.activated.connect(self.elegirModeloEntrenamiento)
        #añadir elementos al comboBox y sus valores asociados
        self.chooseModelComboBox.addItem("KNN",1)
        self.chooseModelComboBox.addItem("Naive Bayes",2)
        self.chooseModelComboBox.addItem("Decision Tree",3)

        #Btn para entrenar el modelo seleccionado
        self.trainModelBtn.clicked.connect(self.entrenarModelo)

        #Btn Insertar Datos Testeo
        self.testingFilesBtn.clicked.connect(self.insertarNoticiasTesteo)

        #Btn Seleccionar Modelo
        self.selectTestModelBtn.clicked.connect(self.elegirModeloTesteo)

        #Btn Ejecutar Testeo
        #self.testBtn.clicked.connect(self.funcionquenoexisteaunjaja)

    #funciones
    #=========

    # abrir dialog window para seleccionar los datos de entrenamiento
    def insertarNoticiasEntrenamiento(self):
        filespaths = self.openDialogBox()
        self.stepTipsField.setPlainText("Seleccionamos los directorios donde tenemos los archivos de texto que utilizaremos para entrenar nuestro modelo.")
        #Noticias
        ingresar_noticias(filespaths, noticias)

        ingresar_noticias("despoblación/*.txt", despoblacion)

        #abrir ventana dialogo para seleccionar archivos
        #CODE HERE

        #cambiar self.procesarTextoBtn a habilitado
        self.procesarTextoBtn.setEnabled(True)


    def procesarTexto(self):
        #cambiar texto en self.stepTipsField
        self.stepTipsField.setPlainText("El preprocesamiento a realizar consta de 4 etapas:\n1. Tokenizar: separar las palabras que componen un texto, obteniendo como resultado una secuencia de tokens.\n2. Normalización: se pasa a minúsculas tdoos los tokens.\n3.Filtrado de stopwords: en esta etapa eliminamos  aquellas palabras con poco valor semántico, denominadas stopwords.\n4.Stemming: extraemos el lexema de los tokens restantes  (un ejemplo sería ‘cas-’ para la palabra ‘casero’)")

        #Procesamiento de texto
        texto_procesado(processed_text_entrenamiento, noticias)


        #Creación de arreglo de clases
        crea_clases(clases, processed_text_entrenamiento, despoblacion)
        #print(clases[len(despoblacion)+10])

    def openDialogBox(self):
        filenames = QFileDialog.getOpenFileNames()
        return filenames[0]

        #cambiar self.trainModelBtn a habilitado
        self.trainModelBtn.setEnabled(True)


    #def elegirModeloEntrenamiento(self,index):
        #tomar valor actual del comboBox
     #   modelSelect = self.chooseModelComboBox.itemData(index)


    def insertarNoticiasTesteo(self):
        filepaths = self.openDialogBox()

        #ingresar noticias
        ingresar_noticias(filepaths, nuevas)

        #Procesamiento de texto
        texto_procesado(processed_text_testeo, nuevas)


    def elegirModeloTesteo(self):
        #abrir ventana de diálogo para seleccionar archivo de modelo
        asdf=1


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

    def entrenarModelo(self,index):
        #cambiar texto en self.stepTipsField
        self.stepTipsField.setPlainText(" Entrenando el modelo seleccionado")

        #tomar valor actual del comboBox
        modelSelect = self.chooseModelComboBox.itemData(index)
        print( "opcion seleccionada:",modelSelect)
        #no existe switch en python (o.o)
        if modelSelect == 1:
            self.entrentrenamientoKnn()

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
