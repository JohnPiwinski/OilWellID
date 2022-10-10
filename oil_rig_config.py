import os

from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        # Define yourself dataset path
        self.data_dir = "./dataset/images"
        self.train_ann = "train.json"
        self.val_ann = "valid.json"
        self.max_epoch = 10

        self.num_classes = 1
