"""
Created on Mon Mar  6 18:47:26 2023

@author: fantasma
"""



import math
import tensorflow as tf
import numpy as np 
import pandas as pd 
#from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from tensorflow.keras import layers, preprocessing, models
import re
# import tensorflow.compat.v2 
import tensorflow_datasets

from bs4 import BeautifulSoup
train_df = pd.read_csv("C:/Users/Fantasma/Documents/REDES NEURONALES/train.csv")
test_df = pd.read_csv("C:/Users/Fantasma/Documents/REDES NEURONALES/test.csv")

# train_df[train_df["target"] == 0]["text"].values[1]
# train_df[train_df["target"] == 1]["text"].values[1]

# count_vectorizer = feature_extraction.text.CountVectorizer()

# ## let's get counts for the first 5 tweets in the data
# example_train_vectors = count_vectorizer.fit_transform(train_df["text"])


# print(example_train_vectors[0].todense().shape)
# print(example_train_vectors[0].todense())

# train_vectors = count_vectorizer.fit_transform(train_df["text"])

# ## note that we're NOT using .fit_transform() here. Using just .transform() makes sure
# # that the tokens in the train vectors are the only ones mapped to the test vectors - 
# # i.e. that the train and test vectors use the same set of tokens.
# test_vectors = count_vectorizer.transform(test_df["text"])

# clf = linear_model.RidgeClassifier()

# scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
# scores

# clf.fit(train_vectors, train_df["target"])          

def clean(txt):
    txt = BeautifulSoup(txt, "lxml").get_text()
    #como son tweets eliminamos la menciones @
    txt = re.sub(r"@[A-Za-z0-9]+",  ' ', txt)
    #eliminamos links 
    txt = re.sub(r"https?://[A-Za-z0-9./]+", ' ', txt)
    #nos quedamos solo con caracteres 
    txt = re.sub(r"[^A-Za-z0-9!?']",' ', txt)
    txt = re.sub(r"[...]", ' ', txt)
    #quitamos espacios vacios
    txt = re.sub(r" +", ' ', txt)
    #retornamos variable
    return txt

#los tarjets


train_df_cl = [clean(txt) for  txt in train_df.text]

train_label = train_df.target.values

test_df_cl = [clean(txt) for txt in test_df.text]
#limpieza de tweet o dati con el def por cada txt dentro de el 

#tokenizacion
#codificando datos para darle un contexto convierte el texto a numeros entendibles para la pc 
#se transformaran de strings a numeros
#construye el tokenizador a un corpus con las librerias llamadas en cada .
tokenizer = tensorflow_datasets.deprecated.text.SubwordTextEncoder.build_from_corpus(
    train_df_cl, target_vocab_size = 2**16
    )
train_inp = [tokenizer.encode(sentence) for sentence in train_df_cl]
#se conviertea identificaadores, lo vuelve numeros en secuencia puras listas de numero 
tokeni = tensorflow_datasets.deprecated.text.SubwordTextEncoder.build_from_corpus(
    test_df_cl, target_vocab_size = 2**16
    )
train_inp = [tokenizer.encode(sentence) for sentence in train_df_cl]
test_inp = [tokeni.encode(sentence) for sentence in test_df_cl]

#al padear agarramos la cadena mas larga como indice y las palabras cortas agrega los 0 
#ya que nos da una lista de secuencia cada secuencia recorre a una de numeros

#padding
MAX_LEN = max([len(sentence) for sentence in train_inp])
train_inp = tf.keras.utils.pad_sequences(train_inp,
                                                 value = 0,
                                                 padding = "post",
                                                 maxlen=MAX_LEN)

MAX_LAN = max([len(sentence) for sentence in test_inp])
test_inp = tf.keras.utils.pad_sequences(test_inp,
                                        value = 0,
                                        padding = "post",
                                        maxlen = MAX_LAN)
#hago la secuencia con el de test


#division de conjunto de datos o alcance de datos 

# train_final = np.delete(train_inp,)
#cracion de modelo de tipo dcnn 

class DCNN(tf.keras.Model):
    #se define el constructor en el init siempre va primero
    def __init__(self, #referencia al objeto que guardara los parametros
                 vocab_size,#tamaño de vocabulario lo dara el tokenizador 
                 emb_dim = 128,#espacios embevidos
                 nb_filters = 70,#numeros de filtros
                 FFN_units = 512,#numero de neuronas
                 nb_classes = 2,#posibles categorias en este caso positiva y negativa
                 dropout_rate = 0.1,#el 10% de las neuronas no transmitiran lo aprendido
                 training = False,#entrenamiento y el overfitng
                 name = "dcnn"):#el nombre
        super(DCNN, self).__init__(name = name)#superclase y agregamos el nombre de la clase con self y nombre de red neuronal
        self.embedding = layers.Embedding(vocab_size,#definicion capa es la emb. para vectores
                                          emb_dim)
        
        self.bigram = layers.Conv1D(filters = nb_filters,#brigrama capa de unidimencional 2 en 2
                                    kernel_size = 2,
                                    padding = "valid",#añade 0 donde no haya datos 
                                    activation = "relu")
        #laopcion de bigrama trigrama etc son la suma de el kernel si es tri es de 3 
        
        self.trigram = layers.Conv1D(filters = nb_filters,
                                     kernel_size = 3,
                                     padding = "valid",
                                     activation = "relu")
        self.fourgram = layers.Conv1D(filters = nb_filters,
                                      kernel_size = 4,
                                      padding = "valid",
                                      activation = "relu")
        self.pool = layers.GlobalMaxPool1D()
        #capa de max polling que nos resume el valor maximo de los filtros establecidos
        
        self.dense_1 = layers.Dense(units = FFN_units, activation ="relu")#despues de cada capa oculta se agreag un dropout
        self.dropout = layers.Dropout(rate = dropout_rate)#dropout agregado para prevenir el overfiting
        if nb_classes == 2:#numero de clases a predecir
            self.last_dense = layers.Dense(units = 1, #predecir algun valor 
                                           activation = "sigmoid")#clasificacion sigmoid
        else:
            self.last_dense = layers.Dense(units = nb_classes,
                                           activation = "softmax")#numero de unidades ogual a las que tenemos con probabilidades reales
            
            
    def call(self, inputs, training):#llamar al modelo
        x= self.embedding(inputs)#funcion embebidaas para inputs
        #filtrados
        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)
        #saco las relaciones y pooleo la info
        
        #concatenacion
        merged = tf.concat([x_1, x_2, x_3], axis =-1)#concatenamos las listas formadas
       #batch_size, 3 * nb_filters esta es la entrada correcta 
        merged = self.dense_1(merged) #capa oculta solo en fase de entrenamiento (train)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)#para binario sig y para mas categorias soft

        return output       #devolvemos      
    
    
    
    #CONFIGURACION DE LA RED NEURONAL 
    
VOCAB_SIZE =  tokenizer.vocab_size #da el tamaño del vocab

EMB_DIM = 200 # se vectoriza a 200 numeros
NB_FILTERS = 100 #numero de filtro de la red neuronal
FFN_UNITS = 256 #numeros de unidades por la capa de fitfoward que tendra la capa oculta 
NB_CLASSES = 2 #conjunto de entrenamiento

DROPOUT_RATE = 0.2 #taza de olvidop sera de un 20%

BACHT_SIZE = 32 #de 32 a 32 textos a overfiting
NB_EPOCHS = 10 # repeticiones de entramiento 

#seccion de entrenamiento

Dcnn = DCNN(vocab_size = VOCAB_SIZE,
            emb_dim = EMB_DIM,
            nb_filters = NB_FILTERS,
            FFN_units = FFN_UNITS,
            nb_classes = NB_CLASSES,
            dropout_rate = DROPOUT_RATE)
#datos renombrados para crear el modelo 


#dos casos de tipo binario y de mas 
if NB_CLASSES == 2:
    Dcnn.compile(loss="binary_crossentropy",
                 optimizer= "adam",
                 metrics=["accuracy"])
    
else: 
    Dcnn.compile(loss = "sparce_categorical_crossentropy",
                 optimizer = "adam", #cuanto mayor es la dimension mayor sera la prediccion 
                 metrics =["sparce_categorical_accuracy"])
    
    
#entrenamiento 

Dcnn.fit(train_inp, #VECTORES DE LOS NUMEROS
         train_label, #AJUSTE DE SALIDAS DE DATOS
         batch_size = BACHT_SIZE, #BATCH PARA CORREGIR ERRORES
         epochs = NB_EPOCHS) #VUELTAS DE EPOCAS

#PREDDICCIONES O PRUEBAS

#prueba = Dcnn.evaluate(test_inp, batch_size= BACHT_SIZE)
Dcnn(np.array([tokenizer.encode("Not a diss song People will take 1 thing and run with it Smh it's an eye opener though He is about 2 set the game ablaze ")]), training = False).numpy()

