import numpy as np
import pandas as pd
import pickle
import os
class Classifior():
  # labels= ['diagonal_2points', 'one_line', 'one_surface']
  def __init__(self,data_path:str,model_path:str):
    self.data_path=data_path
    self.model_path=model_path
    pass
  def remove_useless(self,raw_data):
    data=raw_data.drop(["msg","time(second)"],axis=1)
    rot_data = data
    return rot_data
  def load_data(self,file_name:str):
    data_path=self.data_path+file_name
    temp=pd.read_csv(data_path)
    temp=self.remove_useless(temp) 
    row_data=[]
    for row in range (len(temp)):
      row_data.append(temp.iloc[row].values)
    row_data=np.array(row_data).reshape((1,-1)).squeeze(axis=0)
    df=pd.DataFrame(row_data)
    self.df=df.dropna(axis=0) # drop the unconsistent data long
    return self.df
  def fft_self(self,df):
    data_rows = df.copy()
    # Apply FFT on each column except the last row
    fft_result = data_rows.apply(lambda col: np.abs(np.fft.fft(col)), axis=0)
    return fft_result
  
  def processData(self):
    self.df = self.df.iloc[:42000]
    self.df=self.fft_self( self.df)
    return self.df
  def load_model(self,mlp_name:str,kpca_name:str):
    mlp_path=self.model_path+mlp_name
    kpca_path=self.model_path+kpca_name
    self.model_clf = pickle.load(open(mlp_path, 'rb'))     
    self.kernel_pca = pickle.load(open(kpca_path, 'rb'))    
    pass
  
  def fit(self):
    input=self.df.copy().T
    input=self.kernel_pca.transform(input)
    return self.model_clf.predict(input)
  
  def predict(self, data):
    df = pd.DataFrame(data)  # convert data to df
    input = df.copy().T
    input = self.kernel_pca.transform(input)
    return self.model_clf.predict(input)

  def find_latest_file(self):
      files = os.listdir(self.data_path)
      csv_files = [f for f in files if f.startswith("output_") and f.endswith(".csv")]
      latest_file = max(csv_files, key=lambda x: int(re.search(r'\d+', x).group()))
      return latest_file

  def predict_from_disk(self):
      latest_file = self.find_latest_file()
      print(f"Using the latest file for prediction: {latest_file}")
      self.load_data(latest_file)
      self.processData()
      return self.fit()

  def predict_label(self):
      pred = self.predict()
      labels = ['diagonal_2points', 'one_line', 'one_surface']
      predicted_label = labels[pred[0]]
      print(f"Predicted label: {predicted_label}")
      return predicted_label
  