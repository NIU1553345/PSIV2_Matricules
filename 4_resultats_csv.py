import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import joblib


def imatge_binaritzada(imatge):
    imatge = cv2.cvtColor(imatge, cv2.COLOR_BGR2GRAY)
    _, imatge_otsu = cv2.threshold(imatge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plt.imshow(imatge_otsu, cmap='gray')
    # plt.title('Imatge binària')
    # plt.show()
    return imatge_otsu


def detectar_contorns(imatge_binaria):
    contornos, _ = cv2.findContours(imatge_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imatge_contorns = cv2.cvtColor(imatge_binaria, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(imatge_contorns, contornos, -1, (0, 255, 0), 2)
    # plt.imshow(imatge_contorns)
    # plt.title('Tots els contorns detectats')
    # plt.show() 
    return contornos


def retallar_contorn_matricula(imatge, contorns):
    contorn_matricula = None
    max_area = 0  
    for contorn in contorns:
        x, y, w, h = cv2.boundingRect(contorn)
        area = w * h     
        # Proporció Amplada x Alçada per poder filtrar els contorns que no siguin rectangulars
        aspect_ratio = float(w) / h    
        # Filtrar contorns que tinguin entre 2 i 6 de aspect ratio ( mides arbitraries ) i que tinguin l'area més gran	
        if 2 < aspect_ratio < 6 and area > max_area:
            contorn_matricula = contorn
            max_area = area   
    imatge_resultat = imatge.copy()
    cv2.drawContours(imatge_resultat, [contorn_matricula], -1, (0, 255, 0), 2)    
    plt.imshow(cv2.cvtColor(imatge_resultat, cv2.COLOR_BGR2RGB))
    plt.title('Contorn de la matrícula')
    plt.show()
    return contorn_matricula


def retallar_matricula(imatge, contorn_matricula):
    x, y, w, h = cv2.boundingRect(contorn_matricula)
    matricula = imatge[y:y+h, x:x+w]    
    # plt.imshow(cv2.cvtColor(matricula, cv2.COLOR_BGR2RGB))
    # plt.title('Matrícula')
    # plt.show()
    return matricula


def calcular_lluminositat(img):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lluminositat = np.mean(hls_img[:, :, 1])  # Canal L
    return lluminositat


def eliminar_soroll(imatge,contorns):
    ll=[]
    # min_area = 400 
    min_area = 100
    max_area = 20000  
    altura_imatge, base_imatge = imatge.shape[:2]
    contorn_filtrat = []
    for contorn in contorns:
        x, y, w, h = cv2.boundingRect(contorn)
        lletra = matricula[y:y+h, x:x+w]
        area = cv2.contourArea(contorn)
        l=calcular_lluminositat(lletra)    
        # if min_area < area < max_area and x > 20 and x + w < base_imatge - 20 and y > 20 and y + h < altura_imatge - 20 and h>w:
        if min_area < area < max_area and h>w and x + w < base_imatge - 10:
            contorn_filtrat.append(contorn)
            ll.append(l) 
    if len(contorn_filtrat)>7:
        min1, min2 = sorted(ll)[:2]
        diferencia = abs(min1 - min2)
        if diferencia>10:
            pos= ll.index(min(ll))  
            del contorn_filtrat[pos]
        else:
    # if len(contorn_filtrat)>7:
            contorn_filtrat=contorn_filtrat[1:]  
    return contorn_filtrat


def segmentar_matricula(matricula):    
    matricula_gris = cv2.cvtColor(matricula, cv2.COLOR_BGR2GRAY)
    _, matricula_binaria = cv2.threshold(matricula_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    matricula_binaria = cv2.bitwise_not(matricula_binaria)
    contorns, _ = cv2.findContours(matricula_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    # Descarta els contorns menors de 400 píxels
    # contorns = [contorn for contorn in contorns if cv2.contourArea(contorn) > 400]
    contorns = [contorn for contorn in contorns if cv2.contourArea(contorn) > 100]    
    # Ordenar els contorns de esquerra a dreta
    contorns = sorted(contorns, key=lambda contorn: cv2.boundingRect(contorn)[0])
    contorns= eliminar_soroll(matricula, contorns)
    return contorns


def llegir_matricula(contorns,model,model_n,model_ll):
        resultat1=[]
        resultat2=[]
        numeros=contorns[:4]
        lletres=contorns[-3:]
        for i, contorn in enumerate(numeros):
            x, y, w, h = cv2.boundingRect(contorn)
            numero = matricula[y:y+h, x:x+w]
            # plt.imshow(cv2.cvtColor(numero, cv2.COLOR_BGR2RGB))
            # plt.title(f'Numero {i+1}')
            # plt.axis('off')
            # plt.show()
            cv2.imwrite('tempn.jpg', numero)
            img = tf.keras.preprocessing.image.load_img('tempn.jpg', target_size=(128, 64))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array.flatten().reshape(1, -1)        
            predictions = model_n.predict(img_array)
            predicted_n = predictions[0]         
            resultat1.append(str(predicted_n)) 
        for i, contorn in enumerate(lletres):
            x, y, w, h = cv2.boundingRect(contorn)
            lletra = matricula[y:y+h, x:x+w]
            # plt.imshow(cv2.cvtColor(lletra, cv2.COLOR_BGR2RGB))
            # plt.title(f'Lletra {i+1}')
            # plt.axis('off')
            # plt.show()
            cv2.imwrite('templl.jpg', lletra)
            img = tf.keras.preprocessing.image.load_img('templl.jpg', target_size=(128, 64))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array.flatten().reshape(1, -1)
            predictions = model_ll.predict(img_array)
            predicted_class = predictions[0]         
            resultat1.append(str(predicted_class))   
        for i, contorn in enumerate(contorns):
            x, y, w, h = cv2.boundingRect(contorn)
            lletra = matricula[y:y+h, x:x+w]
            #plt.imshow(cv2.cvtColor(lletra, cv2.COLOR_BGR2RGB))
            # plt.title(f'Lletra {i+1}')
            # plt.axis('off')
            # plt.show()
            cv2.imwrite('templl.jpg', lletra)
            img = tf.keras.preprocessing.image.load_img('templl.jpg', target_size=(128, 64))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array.flatten().reshape(1, -1)        
            predictions = model.predict(img_array)
            predicted_class = predictions[0]              
            resultat2.append(str(predicted_class))      
        return ''.join(resultat1),''.join(resultat2)


def llegir_matriculacnn(contorns,model, model_n, model_ll):
    resultat1 = []
    resultat2 = []
    numeros = contorns[:4]
    lletres = contorns[-3:]   
    for i, contorn in enumerate(numeros):
        x, y, w, h = cv2.boundingRect(contorn)
        numero = matricula[y:y+h, x:x+w]
        cv2.imwrite('tempn.jpg', numero)
        img = tf.keras.preprocessing.image.load_img('tempn.jpg', target_size=(128, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Expandir a batch size 1
        prediccion_numero = model_n.predict(img_array)
        predicted_number = np.argmax(prediccion_numero, axis=1)[0]  # Clase predicha
        resultat1.append(str(predicted_number))
    for i, contorn in enumerate(lletres):
        x, y, w, h = cv2.boundingRect(contorn)
        lletra = matricula[y:y+h, x:x+w]
        cv2.imwrite('templl.jpg', lletra)
        img = tf.keras.preprocessing.image.load_img('templl.jpg', target_size=(128, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediccion_lletra = model_ll.predict(img_array)
        predicted_letter = np.argmax(prediccion_lletra, axis=1)[0]  # Clase predicha
        classes = ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
        resultat1.append(str(classes[predicted_letter]))        
    for i, contorn in enumerate(contorns):
        x, y, w, h = cv2.boundingRect(contorn)
        lletra = matricula[y:y+h, x:x+w]
        #plt.imshow(cv2.cvtColor(lletra, cv2.COLOR_BGR2RGB))
        # plt.title(f'Lletra {i+1}')
        # plt.axis('off')
        # plt.show()
        cv2.imwrite('templl.jpg', lletra)
        img = tf.keras.preprocessing.image.load_img('templl.jpg', target_size=(128, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_letter = tf.argmax(predictions[0]).numpy()        
        ll=['0','1','2','3','4','5','6','7','8','9','B','C','D','F','G','H','J','K','L','M','N','P','R','S','T','V','W','X','Y','Z']
        resultat2.append(str(ll[predicted_letter]))    
    return ''.join(resultat1), ''.join(resultat2)



modelcnn = load_model('model_general2.h5')
modelsvm=joblib.load('svm_model_general.pkl')
modelknn=joblib.load('knn_model_general.pkl')
model_llcnn = load_model('model_ll8.h5')
model_llsvm = joblib.load('svm_model_lletres.pkl')
model_llknn = joblib.load('knn_model_lletres.pkl')
model_ncnn = load_model('model_n8.h5')
model_nsvm = joblib.load('svm_model_n.pkl')
model_nknn = joblib.load('knn_model_numeros.pkl')
carpeta=r"C:\Users\Usuario\Downloads\tallades"
# carpeta=r"C:\Users\Usuario\Downloads\prov"

i=0
for file in os.listdir(carpeta):
    filename = os.path.join(carpeta, file) 
    resultats = {
        'CNN2models': [],
        'CNNmodelgeneral': [],
        'SVM2models': [],
        'SVMmodelgeneral': [],
        'KNN2models': [],
        'KNNmodelgeneral': []
    }
    imatge = cv2.imread(filename)
    imatge_binaria = imatge_binaritzada(imatge)
    contorns = detectar_contorns(imatge_binaria)
    contorn_matricula = retallar_contorn_matricula(imatge, contorns)
    matricula = retallar_matricula(imatge, contorn_matricula)
    matricula_segmentada = segmentar_matricula(matricula)
    
    def probar_modelos(imagen, valor_real):
        
        pred_cnn_parcial, pred_cnn_general = llegir_matriculacnn(imagen,modelcnn, model_ncnn, model_llcnn)
        pred_svm_parcial, pred_svm_general = llegir_matricula(imagen,modelsvm, model_nsvm, model_llsvm)
        pred_knn_parcial, pred_knn_general = llegir_matricula(imagen, modelknn,model_nknn, model_llknn)
    
        resultats['CNN2models'].append({'predicció': pred_cnn_parcial, 'valor_real': valor_real})
        resultats['CNNmodelgeneral'].append({'predicció': pred_cnn_general, 'valor_real': valor_real})
        resultats['SVM2models'].append({'predicció': pred_svm_parcial, 'valor_real': valor_real})
        resultats['SVMmodelgeneral'].append({'predicció': pred_svm_general, 'valor_real': valor_real})
        resultats['KNN2models'].append({'predicció': pred_knn_parcial, 'valor_real': valor_real})
        resultats['KNNmodelgeneral'].append({'predicció': pred_knn_general, 'valor_real': valor_real})
    
    nom = os.path.splitext(os.path.basename(filename))[0]
    probar_modelos(matricula_segmentada, nom)
    
    df_resultats = pd.DataFrame({
        'Valor Real': nom[:7],
        'CNN2models': [r['predicció'] for r in resultats['CNN2models']],
        'CNNmodelgeneral': [r['predicció'] for r in resultats['CNNmodelgeneral']],
        'SVM2models': [r['predicció'] for r in resultats['SVM2models']],
        'SVMmodelgeneral': [r['predicció'] for r in resultats['SVMmodelgeneral']],
        'KNN2models': [r['predicció'] for r in resultats['KNN2models']],
        'KNNmodelgeneral': [r['predicció'] for r in resultats['KNNmodelgeneral']],
    })
    #Guardar els resultats en un csv
    if i==0:
        df_resultats.to_csv('resultats_modelsprova.csv', index=False)
    else:
    
        df_resultats.to_csv('resultats_modelsprova.csv', mode='a', index=False, header=False)
    
    i+=1
    print(f"Prediccions {i} guardads a 'resultats_models.csv'")
print("final")
