#create prediction for the save model
import pickle
import pandas as pd

Pkl_Filename = "Pickle_CART_Model.pkl" 

with open(Pkl_Filename, 'rb') as file:  
    Pickled_Model = pickle.load(file)

print(Pickled_Model)

Xdata = pd.read_pickle("./Xdata.pkl")
Ypredict = Pickled_Model.predict(Xdata)

print(Ypredict)