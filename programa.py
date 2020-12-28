from funciones import ingresar_noticias, texto_procesado, crea_clases, tfid, naive_bayes, decision_tree, string_doc

#Noticias
noticias = []
ingresar_noticias("despoblación/*.txt", noticias)
ingresar_noticias("no_despoblación/*.txt", noticias)

#Procesamiento de texto
processed_text = []
processed_text = texto_procesado(processed_text, noticias)

#Creación de clases
clases = []
clases = crea_clases(clases, processed_text)

#TFIDF
X_traincv = tfid(processed_text)

#Naive Bayes
naive = naive_bayes(X_traincv, clases)

#Decision Tree
tree = decision_tree(X_traincv, clases)

#Prediccion
pred_tree = tree.predict(X_traincv[0])
pred_naive = naive.predict(X_traincv[0])
texto = string_doc(0, noticias)
print("La noticia:\n", texto)
print("\n según Decision Tree es de ", pred_tree)
print("\n según Naive Bayes es de ", pred_naive)

