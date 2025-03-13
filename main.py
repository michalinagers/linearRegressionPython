import pandas as pd  #For handling dataframes
import matplotlib.pyplot as plt  #For plotting graphs
import seaborn as sns  #For enhanced visualizations
from sklearn.model_selection import train_test_split  #To split data
from sklearn.linear_model import LinearRegression  #Linear regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

#Load dataset
Salary = pd.read_csv('/content/drive/MyDrive/Salary_Data.csv')

#Display basic information
print(Salary.head())
print(Salary.info())

#Selecting relevant columns
experience = Salary[['YearsExperience', 'Age', 'Salary']]
print(experience.head())

#Handling missing values
experience = experience.dropna()
print(experience.isnull().sum())   #Confirming no null values remain

#Display statistics
print(experience.describe())

#Data visualization
sns.pairplot(Salary)
sns.heatmap(experience.corr(), annot=True)
plt.show()

#Define feature and target variable
x = experience[['YearsExperience']]
y = experience['Salary']

print(x.shape, y.shape)

#Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Initialize and train the model
model = LinearRegression()
model.fit(x_train, y_train)

#Display intercept and coefficients
print("Intercept:", model.intercept_)
coeff_df = pd.DataFrame(model.coef_, x.columns, columns=['Coefficient'])
print(coeff_df)

#Making predictions
predictions = model.predict(x_test)
print("Predictions:", predictions)

#Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, edgecolor='black', alpha=0.7, color='pink', label='Predicted Points')

#Regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color='red', linewidth=2, label='Regression Line')

#Prediction line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='green', linewidth=2, label='Perfect Prediction')

plt.xlabel('Actual Salary (Y Test)', fontsize=12, weight='bold')
plt.ylabel('Predicted Salary (Y Pred)', fontsize=12, weight='bold')
plt.title('Actual vs Predicted Salary with Regression Line', fontsize=14, weight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()

#Model evaluation
print('MAE:', mean_absolute_error(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))

#User input for prediction
years_experience = float(input("Enter years of experience: "))
input_features = np.array([[years_experience]])
predicted_salary = model.predict(input_features)
print(f"Predicted Salary for {years_experience} years of experience: ${predicted_salary[0]:.2f}")
