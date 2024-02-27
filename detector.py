import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.utils import img_to_array 
from keras.models import load_model
import keras.utils as image
from keras.applications.mobilenet_v2 import preprocess_input
from temporizador import Temporizador
#Temporizador para tiempo por repeticion
temporizador = Temporizador()
tiempo_total = 0.0

# Definicion funcion para calcular angulo
def calcularAngulo(primero,medio,final):
  primero = np.array(primero)
  medio = np.array(medio)
  final = np.array(final)
  
  radianes = np.arctan2(final[1]-medio[1], final[0]-medio[0]) - np.arctan2(primero[1]-medio[1], primero[0]-medio[0])
  angulo = np.abs((radianes * 180.0)/np.pi)
  
  if angulo > 180:
    angulo = 360 - angulo  
    
  return angulo

def calcular_porcentaje(valor, minimo, maximo):    
    if minimo == maximo:
        return 100  # Evita la división por cero si minimo y maximo son iguales
    
    # Calcula el porcentaje
    porcentaje = ((valor - minimo) / (maximo - minimo)) * 100
    if(porcentaje>100): porcentaje = 100
    if(porcentaje<0): porcentaje = 0
    return porcentaje

#Variables para cargar el modelo
#softmax
#modelo = 'F:/EPN/QUINTO SEMESTRE/Inteligencia artificial/PROYECTO SEMESTRAL/Modelo softmax/Modelo_softmax.h5'
#peso = 'F:/EPN/QUINTO SEMESTRE/Inteligencia artificial/PROYECTO SEMESTRAL/Modelo softmax/Pesos_softmax.h5'
#Sigmoid
modelo = 'F:/EPN/QUINTO SEMESTRE/Inteligencia artificial/PROYECTO SEMESTRAL/Modelo sigmoid/Modelo_sigmoid.h5'
peso = 'F:/EPN/QUINTO SEMESTRE/Inteligencia artificial/PROYECTO SEMESTRAL/Modelo sigmoid/Peso_sigmoid.h5'

cnn = load_model(modelo)
cnn.load_weights(peso)

#Variables para tener un espacio negro en pantalla
ancho_pantalla, alto_pantalla = 600, 650
#Variables para dibujar y obtener la forma de conectar las marcas con mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
#Camara
cap = cv2.VideoCapture(0)
cv2.namedWindow('Maestro del press militar', (cv2.WINDOW_NORMAL))
cv2.resizeWindow('Maestro del press militar', ancho_pantalla, alto_pantalla )
#Contar repeticiones
contador = 0
estado = None
#Obvear cara y piernas
CORTAR_PUNTOS = (10,25)  #(desde,hasta)    
MY_CONNECTIONS = frozenset([t for t in mp_pose.POSE_CONNECTIONS if t[0] > CORTAR_PUNTOS[0] and t[1] > CORTAR_PUNTOS[0] if t[0] < CORTAR_PUNTOS[1] and t[1] < CORTAR_PUNTOS[1]])

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
  while cap.isOpened():
    cv2.moveWindow('Maestro del press militar', 30, 30) #Posición pantalla
    ret, frame = cap.read()  
    copia = frame.copy()
    fondo_negro = np.zeros((alto_pantalla,ancho_pantalla,3), np.uint8)
    
    #Recolor imagen para mediapipe a rgb
    imagen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen.flags.writeable = False
    #Hacer la detección de la pose
    results = pose.process(imagen)
    #Recolor a bgr
    imagen.flags.writeable = True
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    
    #Obtenemos el alto y el ancho de lo que captura la camara    
    imagen = cv2.resize(imagen, (ancho_pantalla,480))
    alto, ancho, z = imagen.shape    
    
    #Extraer landmarks
    try:
      
      landmarks = results.pose_landmarks.landmark
      
      #Volver invisible las marcas de la cara y de la cintura para abajo
      contadorProvi = 0
      for landmark in landmarks:
        if(contadorProvi<=CORTAR_PUNTOS[0] or contadorProvi>=CORTAR_PUNTOS[1]):
          landmark.visibility = 0
        contadorProvi+=1
      
      #Extraer coordenadas
      hombroIzquierdo = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
      hombroDerecho = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    
      codoIzquierdo = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
      codoDerecho = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    
      muniecaIzquierda = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
      muniecaDerecha = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
      #Calcular angulo entre las marcas
      anguloIzquierdo = calcularAngulo(hombroIzquierdo,codoIzquierdo,muniecaIzquierda)
      anguloDerecho = calcularAngulo(hombroDerecho,codoDerecho,muniecaDerecha)
      #Visualizar los ángulos
      cv2.putText(fondo_negro, f"{anguloIzquierdo:.2f}", (220,100) ,cv2.FONT_HERSHEY_SIMPLEX,0.4,(138, 227, 255),1,cv2.LINE_4)
      cv2.putText(fondo_negro, f"{anguloDerecho:.2f}", (220,140) ,cv2.FONT_HERSHEY_SIMPLEX,0.4,(138, 227, 255),1,cv2.LINE_4)
      
      # PORCENTAJE Y BARRAS
      pocentaje_izquierdo = calcular_porcentaje(anguloIzquierdo,70,140)
      ancho_lleno_iz = int((pocentaje_izquierdo / 100.0) * 60)
      cv2.rectangle(fondo_negro, (420, 100), (420 + ancho_lleno_iz, 105), (37, 194, 243), -1)
      pocentaje_derecho = calcular_porcentaje(anguloDerecho,70,140)
      ancho_lleno = int((pocentaje_derecho / 100.0) * 60)
      cv2.rectangle(fondo_negro, (420, 140), (420 + ancho_lleno, 145), (37, 194, 243), -1)
      porcentaje_prom = (pocentaje_izquierdo+pocentaje_derecho)/2
      alto_lleno_prom = int((porcentaje_prom / 100.0) * 75)
      cv2.rectangle(fondo_negro, (540, 145), (570,145-alto_lleno_prom), (37, 243, 190), -1)  
      
      #Obtener coordenadas para una nueva imagen de prediccion
      x1, y1 = tuple(np.multiply(codoDerecho,[ancho, alto]).astype(int))
      x1 , y1 = (x1 -30), (y1 -225)
      x2 , y2 = tuple(np.multiply(codoIzquierdo,[ancho, alto]).astype(int))
      x2 , y2 = (x2 +30), (y2 +200)
      if(x1<0):
        x1=0
      if(y1<0):
        y1=0
      imagen_predi = copia[y1:y2,x1:x2]      
      cv2.rectangle(imagen, (x1,y1),(x2,y2), (165,165,165), 1)
      
      #Detección modelo 
      """#SOFTMAX====================
      #Recolor imagen a rgb
      img = cv2.cvtColor(imagen_predi, cv2.COLOR_BGR2RGB)
      img = cv2.resize(imagen_predi, (128,128))    #softmax
      img = cv2.resize(imagen_predi, (200,200)) #Sigmoid    
      xa = image.img_to_array(img)
      xa = np.expand_dims(xa,axis = 0)
      vecto = cnn.predict(xa)
      resultado = vecto[0]
      resultado = np.argmax(resultado)
      resultado = int(resultado)      
      
      #Escribir en la imagen que se muestra el resultado de si es correcto o incorrecto
      predicciontxt = None
      if(resultado == 0):
        predicciontxt = "CORRECTO"
        colorT = (255,0,0)
      else:
        predicciontxt = "INCORRECTO"
        colorT = (0,0,255) 
      
      """
      # Sigmoid
      img = cv2.cvtColor(imagen_predi, cv2.COLOR_BGR2RGB)
      img = cv2.resize(imagen_predi, (128,128))
      x = preprocess_input(img)
      x= np.expand_dims(x, axis=0)
      prediccion = cnn.predict(x)
      print(str(prediccion[0][0]))
      
      #Escribir en la imagen que se muestra el resultado de si es correcto o incorrecto
      predicciontxt = None
      if(prediccion[0][0] <= 0.5):
        predicciontxt = "Correcto"
        colorT = (255,188,0)
      else:
        predicciontxt = "Incorrecto"
        colorT = (0,0,255)       
      
      #Visualizar las predicciones solamente dentro de un rango inicial     
      if(anguloIzquierdo < 120 or anguloDerecho<120):
        cv2.putText(fondo_negro, predicciontxt, (100,45) ,cv2.FONT_HERSHEY_SIMPLEX,0.4,colorT,1,cv2.LINE_4)
        
      #Contador
      if(anguloIzquierdo < 120 and anguloDerecho<120 and prediccion[0][0] <= 0.5 ):        #softmax / resultado == 0
        estado = "inicio"
        tiempo_total += temporizador.tiempo_transcurrido()
        temporizador.iniciar()        
      if(anguloIzquierdo > 145 and anguloDerecho > 145 and estado=='inicio'):
        estado="fin"
        contador+=1       
    except:
      pass
    
    #DIBUJOS DENTRO DE LA PANTALLA NEGRA
    cv2.putText(fondo_negro, "Prediccion:  ", (20,45) ,cv2.FONT_HERSHEY_SIMPLEX,0.4,(37, 194, 243),1,cv2.LINE_4)
    cv2.putText(fondo_negro, "N. repeticiones:  ", (220,45) ,cv2.FONT_HERSHEY_SIMPLEX,0.4,(37, 194, 243),1,cv2.LINE_4)
    cv2.putText(fondo_negro, "Tiempo/rep:  ", (440,45) ,cv2.FONT_HERSHEY_SIMPLEX,0.4,(37, 194, 243),1,cv2.LINE_4)
    cv2.rectangle(fondo_negro, (0,50),(ancho_pantalla,52), (44, 153, 188), -1)
    cv2.putText(fondo_negro, "Tiempo total:  ", (20,80) ,cv2.FONT_HERSHEY_SIMPLEX,0.4,(37, 194, 243),1,cv2.LINE_4)
    cv2.putText(fondo_negro, "Angulo brazo derecho:  ", (220,80) ,cv2.FONT_HERSHEY_SIMPLEX,0.4,(37, 194, 243),1,cv2.LINE_4)
    cv2.putText(fondo_negro, "Angulo brazo izquierdo:  ", (220,120) ,cv2.FONT_HERSHEY_SIMPLEX,0.4,(37, 194, 243),1,cv2.LINE_4)
    cv2.rectangle(fondo_negro, (420,100),(480,105), (44, 153, 188), 1)
    cv2.rectangle(fondo_negro, (420,140),(480,145), (44, 153, 188), 1)
    
    cv2.putText(fondo_negro, str(contador), (330,45) ,cv2.FONT_HERSHEY_SIMPLEX,0.4,(138, 227, 255),1,cv2.LINE_4)
    cv2.putText(fondo_negro, str(temporizador.tiempo_transcurrido()), (525,45) ,cv2.FONT_HERSHEY_SIMPLEX,0.4,(138, 227, 255),1,cv2.LINE_4)
    cv2.putText(fondo_negro, f"{int(tiempo_total)} segundos", (40,100) ,cv2.FONT_HERSHEY_SIMPLEX,0.4,(138, 227, 255),1,cv2.LINE_4)

    #Render detecciones y Obvear conexiones de la cara        
    mp_drawing.draw_landmarks(imagen, results.pose_landmarks, MY_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=2))
    
    if alto < alto_pantalla or ancho < ancho_pantalla:
      border_height = int((alto_pantalla-alto))
      border_width = int((ancho_pantalla-ancho)/2)
      fondo_negro[border_height:alto_pantalla ,border_width:ancho_pantalla-border_width] = imagen
    
    cv2.imshow('Maestro del press militar', fondo_negro)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
