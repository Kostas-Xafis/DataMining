import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors._nearest_centroid import NearestCentroid

bancrupt = 'X65'
df = pd.read_csv('./training_companydata_prepped.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(bancrupt, axis=1), df[bancrupt], test_size=0.2, shuffle=True)

model = NearestCentroid()

# Train the model
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
