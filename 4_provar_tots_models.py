import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import joblib


def imatge_binaritzada(imatge):
    imatge = cv2.cvtColor(imatge, cv2.COLOR_BGR2GRAY)
    _, imatge_otsu = cv2.threshold(imatge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #plt.imshow(imatge_otsu, cmap='gray')
    #plt.title('Imatge binària')
    #plt.show()
    return imatge_otsu


def detectar_contorns(imatge_binaria):
    contornos, _ = cv2.findContours(imatge_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imatge_contorns = cv2.cvtColor(imatge_binaria, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(imatge_contorns, contornos, -1, (0, 255, 0), 2)
    #plt.imshow(imatge_contorns)
    #plt.title('Tots els contorns detectats')
    #plt.show()
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
    #plt.imshow(cv2.cvtColor(imatge_resultat, cv2.COLOR_BGR2RGB))
    #plt.title('Contorn de la matrícula')
    #plt.show()
    return contorn_matricula


def retallar_matricula(imatge, contorn_matricula):
    x, y, w, h = cv2.boundingRect(contorn_matricula)
    matricula = imatge[y:y+h, x:x+w]
    #plt.imshow(cv2.cvtColor(matricula, cv2.COLOR_BGR2RGB))
    #plt.title('Matrícula')
    #plt.show() 
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
        l=calcular_lluminositat(lletra)  # if min_area < area < max_area and x > 20 and x + w < base_imatge - 20 and y > 20 and y + h < altura_imatge - 20 and h>w:
        if min_area < area < max_area and h>w and x + w < base_imatge - 10:
            contorn_filtrat.append(contorn)
            ll.append(l) 
    if len(contorn_filtrat)>7:
        min1, min2 = sorted(ll)[:2]
        diferencia = abs(min1 - min2)
        if diferencia>10:
            pos= ll.index(min(ll))  
            del contorn_filtrat[pos]
        
    if len(contorn_filtrat)>7:
        contorn_filtrat=contorn_filtrat[1:] 
    return contorn_filtrat


def segmentar_matricula(matricula):
    matricula_gris = cv2.cvtColor(matricula, cv2.COLOR_BGR2GRAY)
    _, matricula_binaria = cv2.threshold(matricula_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    matricula_binaria = cv2.bitwise_not(matricula_binaria)
    contorns, _ = cv2.findContours(matricula_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Descarta els contorns menors de 400 píxels
    contorns = [contorn for contorn in contorns if cv2.contourArea(contorn) > 400]
    # Ordenar els contorns de esquerra a dreta
    contorns = sorted(contorns, key=lambda contorn: cv2.boundingRect(contorn)[0])
    contorns= eliminar_soroll(matricula, contorns)
 
    # Crear una figura con subgráficas
    #fig, axs = plt.subplots(1, 7, figsize=(20, 4))  # Ajusta el tamaño según sea necesario
    #axs = axs.ravel()  # Aplana la matriz de ejes para facilitar la indexación

    #for i, contorn in enumerate(contorns):
        #x, y, w, h = cv2.boundingRect(contorn)
        #lletra = matricula[y:y+h, x:x+w]

        # Mostrar la letra en el subplot correspondiente
        #axs[i].imshow(cv2.cvtColor(lletra, cv2.COLOR_BGR2RGB))
        #axs[i].set_title(f'Caracter {i+1}')
        #axs[i].axis('off')  # Oculta los ejes
 
    #plt.tight_layout()  # Ajusta el espacio entre subgráficas
    #plt.show()
    return contorns
 

def llegir_matriculacnn(contorns, model, model_n, model_ll):
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
    return resultat1, resultat2


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
        return resultat1, resultat2


def matriu_confusio(real, pred, titol, classes):
    matriu = confusion_matrix(real, pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriu, annot=True, fmt='d', cmap='Blues', cbar=False,  xticklabels=classes, yticklabels=classes)
    plt.title(f"Matriu de confusió - {titol}")
    plt.xlabel('Predició')
    plt.ylabel('Real')
    plt.show()

def metriques(real, pred):
    hits = 0
    misses = 0
    total_matricules = len(real) // 7 
    for i in range(total_matricules):
        real_group = real[i*7:(i+1)*7]
        pred_group = pred[i*7:(i+1)*7]
        if real_group == pred_group:
            hits += 1
        else:
            misses += 1
    hit_rate = hits / total_matricules
    miss_rate = misses / total_matricules
    total = len(real)
    correctes = sum(1 for r, p in zip(real, pred) if r == p)  # Comptar encerts
    precisio = correctes / total if total > 0 else 0

    print(f"Hits: {hits}")
    print(f"Misses: {misses}")
    print(f"Hit Percentatge: {hit_rate * 100:.2f}%")
    print(f"Miss Percentatge: {miss_rate * 100:.2f}%")
    print(f"Precisió: {precisio}")    
    
####################################################################################################################################################################################
#Hem provat per tots els models CNN, SVM i KNN
model_ll_cnn = load_model('/Users/aina/Desktop/uni/4rt/psiv/repte1/models/CNN/model_CNN_lletres.h5')
model_n_cnn = load_model('/Users/aina/Desktop/uni/4rt/psiv/repte1/models/CNN/model_CNN_num.h5')
model_general_cnn = load_model('/Users/aina/Desktop/uni/4rt/psiv/repte1/models/CNN/model_CNN_general.h5')

model_ll_svm = joblib.load('/Users/aina/Desktop/uni/4rt/psiv/repte1/models/SVM/model_SVM_lletres.pkl')
model_n_svm = joblib.load('/Users/aina/Desktop/uni/4rt/psiv/repte1/models/SVM/model_SVM_num.pkl')
model_general_svm = joblib.load('/Users/aina/Desktop/uni/4rt/psiv/repte1/models/SVM/model_SVM_general.pkl')

model_ll_knn = joblib.load('/Users/aina/Desktop/uni/4rt/psiv/repte1/models/KNN/model_KNN_lletres.pkl')
model_n_knn = joblib.load('/Users/aina/Desktop/uni/4rt/psiv/repte1/models/KNN/model_KNN_num.pkl')
model_general_knn = joblib.load('/Users/aina/Desktop/uni/4rt/psiv/repte1/models/KNN/model_KNN_general.pkl')

real_num = []
real_lletres = []
real_general = []
cnn_pred_num = []
cnn_pred_lletres = []
cnn_pred_general = []
svm_pred_num = []
svm_pred_lletres = []
svm_pred_general = []
knn_pred_num = []
knn_pred_lletres = []
knn_pred_general = []

path = '/Users/aina/Desktop/uni/4rt/psiv/repte1/be'

for fitxer in os.listdir(path):
    if fitxer.endswith(('.png', '.jpg', '.jpeg')):
        caracters = os.path.splitext(fitxer)[0]
        filename = os.path.join(path, fitxer)
        imatge = cv2.imread(filename)
        imatge_binaria = imatge_binaritzada(imatge)
        contorns = detectar_contorns(imatge_binaria)
        contorn_matricula = retallar_contorn_matricula(imatge, contorns)
        matricula = retallar_matricula(imatge, contorn_matricula)
        matricula_segmentada = segmentar_matricula(matricula)

        print('\nComencem els models!')
        pred_cnn_parcial, pred_cnn_general = llegir_matriculacnn(matricula_segmentada, model_general_cnn, model_n_cnn, model_ll_cnn)
        print('Model CNN fet')
        pred_svm_parcial, pred_svm_general = llegir_matricula(matricula_segmentada, model_general_svm, model_n_svm, model_ll_svm)
        print('Model SVM fet')
        pred_knn_parcial, pred_knn_general = llegir_matricula(matricula_segmentada, model_general_knn, model_n_knn, model_ll_knn)
        print('Model KNN fet')

        print('\nCNN: ')        
        print('real :', caracters)
        print('separat :', pred_cnn_parcial)
        print('junt :', pred_cnn_general)
        
        print('\n\nSVM: ')      
        print('real :', caracters)
        print('separat :', pred_svm_parcial)
        print('junt :', pred_svm_general)
        
        print('\n\nKNN: ')     
        print('real :', caracters)
        print('separat :', pred_knn_parcial)
        print('junt :', pred_knn_general)

        real_num.extend([num for num in caracters[:4]])
        real_lletres.extend([lletra for lletra in caracters[-3:]])
        real_general.extend([car for car in caracters])
        cnn_pred_num.extend(pred_cnn_parcial[:4])
        cnn_pred_lletres.extend(pred_cnn_parcial[-3:])
        cnn_pred_general.extend(pred_cnn_general)
        svm_pred_num.extend(pred_svm_parcial[:4])
        svm_pred_lletres.extend(pred_svm_parcial[-3:])
        svm_pred_general.extend(pred_svm_general)
        knn_pred_num.extend(pred_knn_parcial[:4])
        knn_pred_lletres.extend(pred_knn_parcial[-3:])
        knn_pred_general.extend(pred_knn_general)


caracters=['0','1','2','3','4','5','6','7','8','9','B','C','D','F','G','H','J','K','L','M','N','P','R','S','T','V','W','X','Y','Z']
matriu_confusio(real_num, cnn_pred_num, "Model CNN Números", caracters[:9])
matriu_confusio(real_lletres, cnn_pred_lletres, "Model CNN Lletres", caracters[10:])
matriu_confusio(real_general, cnn_pred_general, "Model CNN General", caracters)
matriu_confusio(real_num, svm_pred_num, "Model SVM Números", caracters[:9])
matriu_confusio(real_lletres, svm_pred_lletres, "Model SVM Lletres", caracters[10:])
matriu_confusio(real_general, svm_pred_general, "Model SVM General", caracters)
matriu_confusio(real_num, knn_pred_num, "Model KNN Números", caracters[:9])
matriu_confusio(real_lletres, knn_pred_lletres, "Model KNN Lletres", caracters[10:])
matriu_confusio(real_general, knn_pred_general, "Model KNN General", caracters)

cnn_pred_separat=[]
svm_pred_separat=[]
knn_pred_separat=[]
for i in range(len(cnn_pred_num) // 4):
    cnn_pred_separat.extend(cnn_pred_num[i*4:(i+1)*4])
    cnn_pred_separat.extend(cnn_pred_lletres[i*3:(i+1)*3])
    svm_pred_separat.extend(svm_pred_num[i*4:(i+1)*4])
    svm_pred_separat.extend(svm_pred_lletres[i*3:(i+1)*3])
    knn_pred_separat.extend(knn_pred_num[i*4:(i+1)*4])
    knn_pred_separat.extend(knn_pred_lletres[i*3:(i+1)*3])

print('\nCNN: ')
print('       Model separat')
metriques(real_general, cnn_pred_separat)
print('\n       Model general')
metriques(real_general, cnn_pred_general) 

print('\n\nSVM: ')
print('       Model separat')
metriques(real_general, svm_pred_separat)
print('\n       Model general')
metriques(real_general, svm_pred_general)

print('\n\nKNN: ')
print('       Model separat')
metriques(real_general, knn_pred_separat)
print('\n       Model general')
metriques(real_general, knn_pred_general)

