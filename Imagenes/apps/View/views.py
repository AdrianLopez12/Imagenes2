from django.shortcuts import render
from apps.Logica.ModeloCNN import modeloCNN
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
import json
from django.http import JsonResponse

class Clasificacion():
    def determinarCancer(request):
        return render(request, 'informe.html')
    
    @api_view(["GET", "POST"])
    def predecir(request):
        try:
            imagen = request.FILES["imagen"]  # Obtener la imagen del formulario
            resul = modeloCNN.predecir_etiqueta(imagen)  # Pasar la imagen al método de predicción
        except ValueError as e:
            resul = "Error"
        return JsonResponse({"respuesta": resul})
    
    @csrf_exempt
    @api_view(["GET", "POST"])
    def predecirIOJson(request):
        print(request)
        print("****")
        print(request.body)
        print("****")
        body = json.loads(request.body.decode('utf-8'))
        imagen = str(body["imagen"])
        print(imagen)
        resul = modeloCNN.predecir_etiqueta(imagen)  # Pasar la imagen al método de predicción
        data = {"resultado": resul}
        resp = JsonResponse(data)
        resp["Access-Control-Allow-Origin"] = "*"
        return resp
