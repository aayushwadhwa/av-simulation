import pandas as pd
import os

class Load:
    def __init__(self):
        self.filenames = []
    
    def get_files(self, path, num_trajs):
        self.filenames = [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
        self.filenames = self.filenames[:num_trajs]
    
    def get_from_csv(self, filename):
        df = pd.read_csv(filename)
        df.columns = ["t", "v_leader", "x_leader",
                    "v_follower", "x_follower", "v_human", "x_human", "vref"]
        df.drop("t", axis=1, inplace=True)
        return df
    
    def get_data(self, dir_path, num_trajs):
        print("Getting Data...")
        self.get_files(dir_path, num_trajs)
        df = pd.DataFrame()
        print("Preparing Data...")
        i = 1
        for filename in self.filenames:
            data = self.get_from_csv(filename)
            sas = self.get_sas(data)
            df = df.append(sas)
            if i % 10 == 0:
                print(f'Processed data for {i} file(s)')
            i += 1
        return df
    
    def get_sas(self, data):
        sas = pd.concat([data, data.shift(-1)], axis=1)
        sas = sas.iloc[:-1, :-1]
        return sas