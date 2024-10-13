import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

 
def matriu_confusio(df, model, valor_real, pred):
    vr = list(df[valor_real].values)
    prediccions = list(df[pred].values)
    y_true = []
    y_pred = []
    
    for mat in vr:
        for valor in mat:
            y_true.append(valor)   
    for mat in prediccions:
        for valor in mat:
            y_pred.append(valor)    
            
    labels = sorted(set(y_true + y_pred))    
    cm = confusion_matrix(y_true, y_pred, labels=labels) 
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    plt.figure(figsize=(12, 12))  
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=labels)    
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca(),  values_format=None) 
    plt.title(f"Matriu de Confusió Normalitzada {model}", fontsize=16)
    plt.xlabel("Predicció", fontsize=14)
    plt.ylabel("Real", fontsize=14)   
    plt.xticks(fontsize=12)  
    plt.yticks(fontsize=12) 
    plt.grid(False)      
    plt.savefig(f"C:\\Users\\Usuario\\Downloads\\mc\\matriu_confusio_normalitzada_{model}.png", bbox_inches='tight')    
    plt.show()
    
    

def metriques(real, pred):
    hit = 0
    miss = 0
    for i in range(len(real)):
        if real[i] == pred[i]: 
            hit += 1
        else:
            miss += 1 
    hit_rate = hit / len(real)
    miss_rate = miss / len(real)

    real = [char for matricula in real for char in matricula]
    pred = [char for matricula in pred for char in matricula]
    total = len(real)
    correctes = sum(1 for r, p in zip(real, pred) if r == p)
    precisio = correctes / total if total > 0 else 0

    print(f"Hits: {hit}")
    print(f"Misses: {miss}")
    print(f"Hit Percentatge: {hit_rate * 100:.2f}%")
    print(f"Miss Percentatge: {miss_rate * 100:.2f}%")
    print(f"Precisió: {precisio:.2f}")


file_path = '/Users/aina/Desktop/uni/4rt/psiv/repte1/resultatsfinal.csv'
df = pd.read_csv(file_path)
columnes = df.columns.tolist()
valor_real=columnes[0]
models=columnes[1:]
for model in models:
    print(f"\nModel: {model}")
    matriu_confusio(df, model, valor_real, model)
    metriques(df[valor_real].tolist(), df[model].tolist())

