class StorageNode:
    def __init__(self):
        self.generate_loss = []
        self.disciminator_loss = []
        self.delta = []
    
    def print_last(self):
        print(f"Loss: D: {self.disciminator_loss[-1]}, G: {self.generate_loss[-1]} Delta: {self.delta[-1]}")