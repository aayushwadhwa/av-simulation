import pandas as pd
import os

class data:
    def __init__(self, filename = "./data/data1.csv"):
        self.filename = filename
    
    def process(self):        
        df = pd.read_csv(self.filename)
        df.columns = ["t", "v_leader", "x_leader",
                    "v_follower", "x_follower", "v_human", "x_human"]
        df.drop(["t", "v_leader", "v_follower", "v_human"], axis=1, inplace=True)
        # x = df["x_leader_next"], df["x_follower_next"], df["x_human_next"] = df["x_leader"].shift(-1), df["x_follower"].shift(
            # -1), df["x_human"].shift(-1)
        return df

    def get_data(self):
        df = self.process()
        x = df[:-1]
        y = df.shift(-1)[:-1]
        return (x, y)
    
    def get_normalized_data(self):
        df = self.process()
        x = df[:-1]
        y = df.shift(-1)[:-1]
        return (self.normalize(x), self.normalize(y))
    
    def normalize(self, df):
        return (df - df.mean())/df.std()

class DataPrep:
    def __init__(self):
        self.destdir = ""
        self.filenames = []
    
    def get_files(self, train=True, validation=False):
        if validation:
            self.destdir = './data/validation_data'
        elif train:
            self.destdir = './data/training_data'
        else:
            self.destdir = './data/test_data'
            # self.filenames = [os.path.join(self.destdir,f) for f in os.listdir(self.destdir) if os.path.isfile(os.path.join(self.destdir,f))]
        self.filenames = [os.path.join(self.destdir,f) for f in os.listdir(self.destdir) if os.path.isfile(os.path.join(self.destdir,f))]
        # print(f"{'Training Mode' if train else 'Testing Mode'}, got f{len(self.filenames)} file(s)")

    def get_for_csv(self, filename):
        df = pd.read_csv(filename)
        df.columns = ["t", "v_leader", "x_leader",
                    "v_follower", "x_follower", "v_human", "x_human"]
        df.drop("t", axis=1, inplace=True)
        # x = df.iloc[:, :-2]
        x = df
        # y = df.iloc[:, -2:].shift(-1)
        y = df.shift(-1)
        return x[:-1],y[:-1]
    
    def get_data(self, train=True, validation=False):
        print("Getting Data...")
        self.get_files(train=train, validation=validation)
        x = pd.DataFrame()
        y = pd.DataFrame()
        print("Preparing Data...")
        i = 1
        for filename in self.filenames:
            temp_x, temp_y = self.get_for_csv(filename)
            temp_x = self.normalize(temp_x)
            temp_y = self.normalize(temp_y)
            x = x.append(temp_x)
            y = y.append(temp_y)
            if i % 20 == 19:
                print(f'Processed data for {i} file(s)')
            i += 1
        # df.columns = ["t", "v_leader", "x_leader",
                    # "v_follower", "x_follower", "v_human", "x_human"]
        return x, y
    
    def normalize(self, df):
        # return (df - df.mean())/df.std()
        return df

# d = DataPrep()
# d.get_data()