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

        self.trainingAddFilesBtn.clicked.connect(self.insertarNoticiasEntrenamiento)
        self.testingFilesBtn.clicked.connect(self.insertarNoticiasTesteo)

        #self.trainingModelNB.clicked.connect(self.entrenamientoNaiveBayes)
        #self.trainingModelAD.clicked.connect(self.entrenamientoArbolDecision)
        #self.trainingModelKnn.clicked.connect(self.entrenamientoKnn)


    def insertarNoticiasEntrenamiento(self):
        filespaths = self.openDialogBox()
        
        #Noticias
        ingresar_noticias(filespaths, noticias)
            
        ingresar_noticias("despoblación/*.txt", despoblacion)

        #Procesamiento de texto
        texto_procesado(processed_text_entrenamiento, noticias)


        #Creación de arreglo de clases
        crea_clases(clases, processed_text_entrenamiento, despoblacion)
        #print(clases[len(despoblacion)+10])

    def openDialogBox(self):
        filenames = QFileDialog.getOpenFileNames()
        return filenames[0]

    def insertarNoticiasTesteo(self):
        filepaths = self.openDialogBox()
        
        #ingresar noticias
        ingresar_noticias(filepaths, nuevas)

        #Procesamiento de texto
        texto_procesado(processed_text_testeo, nuevas)

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





#inicializar app
app=QtWidgets.QApplication(sys.argv)
#crear instancia clase initial
mainwindow=initial()


#Stack 
widget=QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.show()
app.exec()
