import cv2
import os 
import numpy as np 

dataPath = 'C:/Users/ANDRES/Desktop/Proyecto/Data'
peopleList = os.listdir(dataPath)
print('lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Leyendo imagenes')

	for fileName in os.listdir(personPath):
		print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
		image = cv2.imread(personPath+'/'+fileName,0)
		cv2.imshow('image',image)
		cv2.waitKey(10)
	label = label+1

cv2.destroyAllWindows

face_recognizer = cv2.face.EigenFaceRecognizer_create()

#Entrenar 
print("Entrenando")
face_recognizer.train(facesData, np.array(labels))

#Almacenar
face_recognizer.write('modeloEigenFace.xml')
print("Modelo almacenado")