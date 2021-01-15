import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi

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
#inicializar app
app=QtWidgets.QApplication(sys.argv)
#crear instancia clase initial
mainwindow=initial()
#Stack 
widget=QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.show()
app.exec()
