from sklearn.feature_extraction.text import TfidfVectorizer
from funciones import ingresar_noticias, texto_procesado, crea_clases, tfid, naive_bayes, decision_tree, string_doc, prediccion, ramdonforest, tfid_fit

#TFIDF Vectorizer
cv = TfidfVectorizer()

#Noticias
noticias = []
ingresar_noticias("despoblación/*.txt", noticias)
ingresar_noticias("no_despoblación/*.txt", noticias)

#Noticias de prueba
nuevas = []
ingresar_noticias("unlabeled/*.txt", nuevas)

#Procesamiento de texto
processed_text = []
processed_text = texto_procesado(noticias)
processed_text_nuevas = texto_procesado(nuevas)

#Creación de arreglo de clases
clases = []
clases = crea_clases(clases, processed_text)

#TFIDF_noticias
X_traincv = tfid(processed_text, cv)

#TFIDF_prueba
X_testcv = tfid_fit(processed_text_nuevas, cv)

#Modelos
naive = naive_bayes(X_traincv, clases)
tree = decision_tree(X_traincv, clases)
random_forest = ramdonforest(X_traincv, clases)

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

