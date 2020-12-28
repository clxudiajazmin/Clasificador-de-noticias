from funciones import ingresar_noticias, texto_procesado, crea_clases, tfid, naive_bayes, decision_tree, string_doc, prediccion, ramdonforest

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

#Modelos
naive = naive_bayes(X_traincv, clases)
tree = decision_tree(X_traincv, clases)
random_forest = ramdonforest(X_traincv, clases)

#Prediccion
pred_tree = prediccion(tree, X_traincv[400])
pred_naive = prediccion(naive, X_traincv[400])
pred_random = prediccion(random_forest, X_traincv[400])

noticia = string_doc(400, noticias)

print("La noticia:\n", noticia)
print("\n según Decision Tree es de ", pred_tree)
print("\n según Naive Bayes es de ", pred_naive)
print("\n según Random Forest es de ", pred_random)

