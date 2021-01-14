from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from funciones import ingresar_noticias, texto_procesado, crea_clases, tfid, naive_bayes, decision_tree, \
    string_doc, prediccion, ramdonforest, tfid_fit, test_score, matrizconf, datos_test, accuracy

#TFIDF Vectorizer
cv = TfidfVectorizer()


#Noticias
noticias = []
ingresar_noticias("despoblación/*.txt", noticias)
ingresar_noticias("no_despoblación/*.txt", noticias)

despoblacion = []
ingresar_noticias("despoblación/*.txt", despoblacion)



nuevas = []
ingresar_noticias("unlabeled/*.txt", nuevas)

#Procesamiento de texto
processed_text = []
processed_text = texto_procesado(noticias)
processed_text_nuevas = texto_procesado(nuevas)

#Creación de arreglo de clases
clases = []
clases = crea_clases(clases, processed_text, despoblacion)

#TFIDF_noticias
X_traincv = tfid(processed_text, cv)

#Partición de datos
X_train, X_test, Y_train, Y_test = train_test_split(X_traincv, clases, test_size=0.15, random_state=324)

#Modelos
naive = naive_bayes(X_train, Y_train)
tree = decision_tree(X_train, Y_train)
random_forest = ramdonforest(X_train, Y_train)
print("####################### Test Score ##############################\n")
print("Naive Bayes: ")
test_score(naive, X_train, Y_train)
print("\nDecision Tree: ")
test_score(tree, X_train, Y_train)
print("\nRandom Forest: ")
test_score(random_forest, X_train, Y_train)

#Creamos los datos a testear
Y_true_naive, Y_pred_naive = datos_test(Y_test, naive, X_test)
Y_true_tree, Y_pred_tree = datos_test(Y_test, tree, X_test)
Y_true_random, Y_pred_random = datos_test(Y_test, random_forest, X_test)

#Datos de los modelos
print("###################### Accuracy ###############################\n")
print("Naive Bayes: ")
accuracy(Y_true_naive,Y_pred_naive)
print("\nDecision Tree: ")
accuracy(Y_true_tree, Y_pred_tree)
print("\nRandom Forest: ")
accuracy(Y_true_random, Y_pred_random)
print("\n###################### Matriz de confusion ###############################\n")
print("Naive Bayes: ")
matrizconf(Y_true_naive,Y_pred_naive)
print("\nDecision Tree: ")
matrizconf(Y_true_tree, Y_pred_tree)
print("\nRandom Forest: ")
matrizconf(Y_true_random, Y_pred_random)

#TFIDF_prueba
X_testcv = tfid_fit(processed_text_nuevas, cv)



#Predicción
pred_tree = prediccion(tree, X_testcv[0])
pred_naive = prediccion(naive, X_testcv[0])
pred_random = prediccion(random_forest, X_testcv[0])

#Impresión de noticia
noticia = string_doc(0, nuevas)

print("La noticia:\n", noticia)
print("\n según Decision Tree es de ", pred_tree)
print("\n según Naive Bayes es de ", pred_naive)
print("\n según Random Forest es de ", pred_random)
print("\n", nuevas[0])

