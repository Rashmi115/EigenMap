from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def build_model():
    # Load the iris dataset
    iris = load_iris()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Create a Logistic Regression model with multinomial (softmax) loss
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Compute the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    confusionMatrix = confusion_matrix(y_test, y_pred)
    
    return {'accuracy': accuracy, 
            'precision': precision, 
            'recall': recall, 
            'f1_score': f1, 
            'confusion_matrix': confusionMatrix.tolist()
            }