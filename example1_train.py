import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import mlflow.models
from mlflow.models.signature import infer_signature

 
# Set up MLflow experiment
mlflow.set_experiment("iris_classification")
 
iris=load_iris()
# print(iris)
print(iris.data) #features
print("********")
print(iris.target) 

x_train,x_test,y_train,y_test= train_test_split(

    iris.data,
    iris.target,
    test_size=0.2,
    random_state=70

)

# print("x_train",x_train) 
# print("y_train",y_train) 
# print("x_test",x_test) 
# print("y_test",y_test) 

with mlflow.start_run():
    model= LogisticRegression(max_iter=200) #Train model
    model.fit(x_train,y_train) #fir the model


    preds = infer_signature(x_train, model.predict(x_test)) #make prediction and calculate accuracy
    #remaning 30 samples 
    #preds are the predicted tagets (0/1/2) for the test set
    acc=accuracy_score(y_test,preds)

    mlflow.log_param("Model_types","Logistic_regression")
    mlflow.log_metric("accuracy",acc)
    mlflow.sklearn.log_model(model,name="model")

    print(f"run logged to Mlflow Accuracy:{acc:3f}")