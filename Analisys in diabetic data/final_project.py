"""
Universidade Federal Fronteira Sul
Tópicos em Aprendizado de Máquina
2018/1
Ivair Puerari

Tarefa Final

P/ Compilar:
python final_project.py diabetic_data.csv
	
"""

import numpy as np
import pandas as pd
import sys
import re
from sklearn.preprocessing import LabelEncoder
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve



if __name__ == "__main__":
    ## Inicialmente é realizado a leitura de dados
    ## E carregado para um objeto pandas dataframe
    try:
        data = pd.read_csv(sys.argv[1],delimiter=',')
    except:
        print('Could not open', sys.argv[1])
        exit(0)
    
    ## Analise dos dados


    ## Verifica-se as labels do conjunto de dados
    ## para visualização e procedimentos futuros
    
    #labels = data.columns.values
    #print(labels)

    ## Verifica-se as dimensoes do conjunto de dados
    m,d= data.shape
    
    ## Os dados vem com valores nulls
    ## Só que, quando carregados, possuem valores como '?' que o python ira interpretar
    ## como um valor existente. 
    ## Para tratar esse problema foi realizado a troca desses valores,
    ## valores com '?' foram substituidos por valores np.nan.
    ## É possivel verificar que há também valores None, mas,
    ## em leitura da descrição dos dados, é possivel verificar que None
    ## assume valor não mensurado e assim,  para valores None, foi realizado
    ## a substituição pela string 'Not meansured' para melhor visualização

  
    data = data.replace('?',np.nan)
    data = data.replace('None','Not Meansured')
    ## Calcula-se os valores null e os guarda em uma serie
    t = data.isnull().sum()
   

    ## Aqui, Realiza-se o tratamento das features que possuem valores np.nan
    ## Instacia-se um dict, onde ira salvar a chave, sendo a label da feature , e o valor, sendo o valor a ser inserido
    ## como troca ao np.nan
    change = {}
    for cnt,v in t.items():
        ## Realiza o calculo da proporção de valores nan em cada feature
        p = (v/m)
        if(p > 0.5): ## Se a feature tiver seus dados contendo 50% dos valores nan, exclui-se a coluna
            data = data.drop(cnt, axis = 1)
        elif(p < 0.5 and p > 0.25):  ## Se a feature contém entre 25 a 50% dos dados np.nan o valor atribuido é 'Not Meansured'
            change[cnt] = 'Not Meansured'    
        elif(p < 0.25 and v > 0): # Se valores np.nan esta em uma faixa de até 25%, o valor com maior ocorrencia na coluna 
            a = data[cnt].value_counts() 
            change[cnt] =  a.idxmax() # sera o valor atribuido a np.nan
    ## Esse tratamento foi realizado, como objetivo, em manter a maior quantidade de originalidade dos dados 
    ## utilizando-se de valores disponiveis

    ## Visualiza-se as features e valores de troca
    #print(change)

    ## É realizado a troca dos valores contidos no dicionario change para cada coluna do dataset 
    new_data = data.fillna(change)

    ## Aqui, realiza-se a discretização de features que contém valores do tipo Nominal(Alphanumericos)
    ## Utilizo a função labelEncoder() do preprocessing que tranforma os valores para valores categorizados
    for cnt,v in enumerate(new_data.columns.values):
        ## Presiva verificar se coluna é do tipo object e se possui valores alphanumericos
        if(new_data[v].dtype == object and map(lambda x: x.isalnum(), new_data[v])):   
            enc = LabelEncoder()
            cat_labels = enc.fit_transform(new_data[v])  
            new_data[v] = cat_labels
    ## Assim o novo conjunto de dados esta discretizado e sem valores nulls
    
    ## Proximo passo foi verificado colunas com apenas um valor, ou seja, colunas com todos os valores duplicados
    ## pois parecem desnecessaria para o conjunto de dados, pois não diferem para nenhum exemplo.     
    ## E também features com dois valores, mais com valores muito mais predominantes para um só valor
    ## considerados outliers
    for cnt,v in enumerate(new_data.columns.values):
        ## verifica ocorrencia de valores para a coluna
        dpc =  new_data[v].value_counts()
        ## Se existir apenas um valor é excluido para todas as linhas da coluna
        if(len(dpc) == 1):
            new_data = new_data.drop(v, axis = 1)
        ## Possui dois valores, mas a maior predominancia é para apenas um valor com menos de 5% de relevancia sobre o total 
        if(len(dpc) == 2 and dpc.loc[1] < 50):
            new_data = new_data.drop(v, axis = 1)       


    ## Separa em exemplos e labels
    ylabels = new_data['readmitted']
    Xlabels = new_data.drop('readmitted', axis = 1)
    labels = Xlabels.columns.values
    ## Cria duas arrays um para exemplos e outro para os labels
    X = np.array(Xlabels)
    y = np.array(ylabels)

    ## 1. Identifique os 10 melhores atributos para a criação do modelo (por exemplo, utilizando o SelectKBest).                                                         
    skb = feature_selection.SelectKBest(k=10).fit(X,y)
    index = skb.get_support(indices = True)
    features = []
    for i in index:
        features.append(labels[i])
    
    ## Divisão dos dados, 90% para treino e 10% para teste 
    X = new_data[features]
    X = np.array(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.10, stratify = y)
    
    ## 2. Selecione 3 algoritmos classificadores para a atividade. 
    ## Logistic Regression
    ## Random forest
    ## KNeighbors
    
    ## Lista  de modelos de classificadores que serão utilizados na aplicação
    models = [LogisticRegression(),RandomForestClassifier(),KNeighborsClassifier()]
    ## Quantidade de modelos
    N = 3
    
    ## 3. Define um conjuno de hiper parâmetros para os modelos 
    ## (selecione pelo menos 3 hiper parâmetros) e defina pelo menos 2 valores para cada hiper parâmetro.
    
    #Lista de hiper parâmetros que irão ser utlizados no gridsearchCV
    hyper = []
    
    ## Logistic Regression
    C=np.logspace(0, 7, 3) 
    penalty=['l1', 'l2']
    class_weight = [None,'balanced']

    param_grid = {
    "penalty": penalty,
    "C": C,
    "class_weight":class_weight
    }
    hyper.append(param_grid)
    
    ## Random forest
    criterion = ['gini','entropy']
    n_estimators = np.random.randint(low=100, high=150, size=3)
    max_features = np.random.randint(low=1, high=7, size=3)
    
    param_grid = {
    "criterion": criterion,
    "n_estimators": n_estimators,
    "max_features":max_features
    }
    hyper.append(param_grid)

    ## KNeighbors
    n_neighbors =np.random.randint(low=1, high=7, size=3)
    weights = ['uniform','distance']
    p = [1,2]

    param_grid = {
    "n_neighbors": n_neighbors,
    "weights": weights,
    "p":p
    }
    hyper.append(param_grid)
 
   
    for i in range(N):
        ## 4. Utilizando o GridSearchCV encontre a melhor combinação de parâmetros para os 3 classificadores 
        ## (utilize 5 fatias para a validação cruzada).       
        grid_search = GridSearchCV(estimator=models[i],param_grid = hyper[i],cv = 5)
        grid_search.fit(Xtr,ytr)
        print(grid_search.best_params_)

        ## 5. Rode os classificadores com um novo conjunto (subconjunto do original) ainda não utilizado (cerca de 10%).
        yhat = grid_search.predict(Xte)

        ## 6. Utilize o método classification_report do pacote metrics para apresentar a performance dos classificadores.
        print(classification_report(yte,yhat))
        print('-'*50)
