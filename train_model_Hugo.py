# Databricks notebook source
# MAGIC %sql
# MAGIC 
# MAGIC select avg(radiant_win) from sandbox_apoiadores.abt_dota_pre_match

# COMMAND ----------

# DBTITLE 1,Imports
#import das libs
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics

import mlflow

#import dos dados
sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match")
df = sdf.toPandas()

# COMMAND ----------

#exibe o uso de memoria do data frame pandas
df.info(memory_usage='deep')

# COMMAND ----------

# DBTITLE 1,Definição das variaveis
target_column = 'radiant_win'
id_column = 'match_id'

features_columns = list( set(df.columns.tolist()) - 
set([target_column,id_column]))

y = df[target_column]
X = df[features_columns]

X

# COMMAND ----------

# DBTITLE 1,Separação em base de treino e teste
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                     y, 
                                                                     test_size=0.2, 
                                                                     random_state=42)
print("numero x train: ", X_train.shape[0])
print("numero x test: ", X_test.shape[0])
print("numero y train: ", y_train.shape[0])
print("numero y test: ", y_test.shape[0])


# COMMAND ----------

# DBTITLE 1,Setup do experimento
mlflow.set_experiment("/Users/hugo.castaldelli@unesp.br/dota-UNESP-Hugo")

# COMMAND ----------

# DBTITLE 1,Run do experimento
with mlflow.start_run():
    mlflow.sklearn.autolog()
    model = ensemble.MultiTaskElasticNet(n_estimators=100, learning_rate=0.7)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    print("Acuracia em treino:", acc_train)

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    acc_test = metrics.accuracy_score(y_test, y_test_pred)
    print("Acuracia em teste:", acc_test)

    


# COMMAND ----------


