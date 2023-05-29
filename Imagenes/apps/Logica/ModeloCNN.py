import pickle
import numpy as np
from PIL import Image
from tensorflow import keras
import io
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile

class modeloCNN:
    """Clase modelo Preprocesamiento Y CNN"""

    @staticmethod
    def cargar_pipeline(pipeline_file):
        # Cargar el pipeline preprocesado desde el archivo pickle
        with open(pipeline_file, 'rb') as f:
            pipeline = pickle.load(f)

        X_train = pipeline['X_train']
        X_test = pipeline['X_test']
        y_train = pipeline['y_train']
        y_test = pipeline['y_test']
        label_encoder = pipeline['label_encoder']

        # Filtrar los objetos None en X_train
        X_train = [x for x in X_train if x is not None]

        # Comprobar el número de muestras de entrenamiento
        num_samples = len(X_train)
        if num_samples < 1:
            raise ValueError("No hay suficientes muestras de entrenamiento.")

        # Convertir las listas de imágenes en un arreglo NumPy
        X_train = np.array(X_train)
        X_test = np.array(X_test)

        return X_train, X_test, y_train, y_test, label_encoder

    @staticmethod
    def cargar_modelo(model_file):
        # Cargar el modelo de red neuronal CNN
        model = keras.models.load_model(model_file)
        return model

    @staticmethod
    def preprocesar_imagen(image):
        # Redimensionar la imagen al tamaño requerido por el modelo
        image = Image.open(image)
        image = image.resize((512, 512))

        # Convertir la imagen en un arreglo NumPy y normalizar los valores de píxeles
        image_array = np.array(image) / 255.0

        # Expandir las dimensiones del arreglo para que coincida con el formato de entrada del modelo
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    @staticmethod
    def predecir_etiqueta(image):
        # Ruta del archivo pickle del pipeline preprocesado
        pipeline_file = 'Recursos/pipeline.pickle'

        # Ruta del archivo del modelo de red neuronal CNN
        model_file = 'Recursos/modeloRedCNN.h5'

        # Cargar el pipeline y el modelo
        X_train, X_test, y_train, y_test, label_encoder = modeloCNN.cargar_pipeline(pipeline_file)
        model = modeloCNN.cargar_modelo(model_file)

        # Realizar el preprocesamiento de la imagen
        preprocesada = modeloCNN.preprocesar_imagen(image)

        # Realizar la predicción utilizando el modelo cargado
        predicciones = model.predict(preprocesada)
        etiqueta_predicha = label_encoder.inverse_transform(np.argmax(predicciones, axis=1))[0]
        resp = modeloCNN.obtener_diagnostico(etiqueta_predicha)
        return resp

    @staticmethod
    def obtener_diagnostico(variable):
        if 'A' in variable:
            return 'adenocarcinoma'
        elif 'B' in variable:
            return 'carcinoma de células pequeñas'
        elif 'E' in variable:
            return 'carcinoma de células grandes'
        elif 'G' in variable:
            return 'carcinoma de células escamosas'
        else:
            return 'Diagnóstico desconocido'
