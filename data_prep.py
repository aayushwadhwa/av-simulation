import pandas as pd
import os

class DataPrep:
    def __init__(self):
        self.destdir = ""
        self.filenames = []
    
    def get_files(self):
        self.destdir = './data/training_data'
        self.filenames = [os.path.join(self.destdir,f) for f in os.listdir(self.destdir) if os.path.isfile(os.path.join(self.destdir,f))]
        self.filenames = self.filenames[0:20]
    
    def get_from_csv(self, filename):
        df = pd.read_csv(filename)
        df.columns = ["t", "v_leader", "x_leader",
                    "v_follower", "x_follower", "v_human", "x_human"]
        df.drop("t", axis=1, inplace=True)
        return df
    
    def get_data(self):
        print("Getting Data...")
        self.get_files()
        df = pd.DataFrame()
        print("Preparing Data...")
        i = 1
        for filename in self.filenames:
            temp_x = self.get_from_csv(filename)
            df = df.append(temp_x)
            if i % 20 == 19:
                print(f'Processed data for {i} file(s)')
            i += 1
        return df
    
    def normalize(self, df):
        # return (df - df.mean())/df.std()
        return df
