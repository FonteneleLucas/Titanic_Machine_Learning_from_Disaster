import csv
import pandas as pd
from sklearn import svm

data = pd.read_csv('train.csv')

X_train = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Cabin']].values
y_train = data['Survived'].values

clf = svm.SVC(kernel='linear')

# Treina o classificador
clf.fit(X_train, y_train)

# Realiza a predição
teste_csv = pd.read_csv('test.csv')
teste_entrada = teste_csv[['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Cabin']].values

# Abre o arquivo de saída
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['PassengerId', 'Survived'])
    
    # Para cada entrada no arquivo de teste, realiza a predição e adiciona ao arquivo de saída
    for i in range(len(teste_entrada)):
        novo_no = [teste_entrada[i][1],teste_entrada[i][2],teste_entrada[i][3],teste_entrada[i][4],teste_entrada[i][5],teste_entrada[i][6]]
        survived = clf.predict([novo_no])[0]
        writer.writerow([int(teste_entrada[i][0]), survived])
