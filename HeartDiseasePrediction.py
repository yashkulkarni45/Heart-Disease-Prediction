# heart disease prediction model using random forest classifier
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
# import pickle
# Load the data
df = pd.read_csv('./content/heart.csv')
# Display the first five rows of the data
df.head()
# Check the shape of the data
df.shape
# Check the data types
df.dtypes
# Check for missing values
df.isnull().sum()
# Check the class distribution
df['target'].value_counts()
# Separate features and target
X = df.drop('target', axis=1)
y = df['target']
# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Check the shape of training set and test set
print(X_train.shape, X_test.shape)
# Create a random forest classifier
rf_clf = RandomForestClassifier(n_estimators=500, random_state=42)
# Train the model
rf_clf.fit(X_train, y_train)
# Function to print the model's performance

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
# Print the model's performance
print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
# Plot the ROC curve
#y_pred_prob = rf_clf.predict_proba(X_test)[:,1]
#fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#plt.plot([0,1], [0,1], 'k--')
#plt.plot(fpr, tpr, label='Random Forest')
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Random Forest ROC Curve')
#plt.show()
# Print the AUC score

# Function to get the user's input
def get_input():
  age = int(input('Enter your age: '))
  gender = input('Enter your gender (0 for Male/1 for Female): ')
  cp = int(input('Enter your chest pain type (0-3): '))
  trestbps = int(input('Enter your resting blood pressure(mm/Hg): '))
  chol = int(input('Enter your cholesterol(mg/dl): '))
  fbs = int(input('Enter your fasting blood sugar (0 for <120 mg/dL/1 for >120 mg/dL): '))
  restecg = int(input('Enter your resting electrocardiographic results (0-2): '))
  thalach = int(input('Enter your maximum heart rate achieved: '))
  exang = int(input('Enter your exercise induced angina (0/1): '))
  oldpeak = float(input('Enter your ST depression induced by exercise relative to rest: '))
  slope = int(input('Enter the slope of the peak exercise ST segment (0-2): '))
  ca = int(input('Enter the number of major vessels (0-3) colored by fluoroscopy: '))
  thal = input('Enter your thalassemia (normal(1)/fixed defect(2)/reversable defect(3)): ')

  # Map the input to the appropriate format
  data = {
    'age': [age],
    'sex': [gender],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
  }
  input_df = pd.DataFrame(data)
  
  # Preprocess the data (if necessary)
  # (This may include scaling, encoding, etc.)
  
  return input_df

# Get the user's input
input_df = get_input()

# Use the model to make a prediction
prediction = rf_clf.predict(input_df)[0]

# Display the prediction
if prediction == 0:
  print('Wohoo! You dont have any heart disease')
else:
  print('Unfortunately, you have a heart disease.Please consult a specialist.\nFollowing are some of the specialists we recommend according to your place of stay :\n1.Yash Kulkarni-Pune')
  print('2.Harshal Kullarkar-Nagpur')