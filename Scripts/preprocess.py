import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    #lets fill null data
    df["Gender"]=df["Gender"].fillna(df["Gender"].mode()[0])
    df["Married"]=df["Married"].fillna(df["Married"].mode()[0])
    df["Dependents"]=df["Dependents"].fillna(df["Dependents"].mode()[0])
    df["Self_Employed"]=df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
    df["Loan_Amount_Term"]=df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
    df["LoanAmount"]=df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Credit_History"]=df["Credit_History"].fillna(df["Credit_History"].mean())

    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)


    #Encode the catagorical data
    df.replace({'Gender':{'Male':1, 'Female':0},
                'Married':{'Yes':1, 'No':0},
                'Education':{'Graduate':1,'Not Graduate':0},
                'Self_Employed':{'Yes':1,'No':0},
                'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},
                'Loan_Status':{'Y':1,'N':0}}, inplace=True)
    return df
def scale_data(X_train, X_test):
    scaler=StandardScaler()
    Xtrain_scaled=scaler.fit_transform(X_train)
    xtest_scaled=scaler.transform(X_test)
    return Xtrain_scaled, xtest_scaled, scaler