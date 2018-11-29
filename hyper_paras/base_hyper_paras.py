"""
This is base hyper-parameters, you can modify them here.
"""


class BaseHyperparameters(object):
    def __init__(self):
        # self.N_FEATURES = [210, 160, 3]  # Without cropping and stack.
        self.N_FEATURES = [80, 80, 4]  # With cropping and stack.
        self.N_ACTIONS = 4
        self.IMAGE_SIZE = 80
        self.model = 'null'

        self.MAX_EPISODES = 5000  # 150000 : 500
        self.LEARNING_RATE = 0.0001
        self.INITIAL_EXPLOR = 0  # 0 : 0.5
        self.FINAL_EXPLOR = 0.9
        self.FINAL_EXPLOR_FRAME = 4000  # 100000 : 450
        self.DISCOUNT_FACTOR = 0.99
        self.MINIBATCH_SIZE = 64  # 32 : 8
        self.REPLY_START_SIZE = 1000  # 10000 : 100
        self.REPLY_MEMORY_SIZE = 500
        self.TARGET_NETWORK_UPDATE_FREQUENCY = 800  # 2000 : 150

        # log and output
        self.WEIGHTS_SAVER_ITER = 1000  # 2000 : 200
        self.OUTPUT_SAVER_ITER = 1000  # 1000 : 100
        self.OUTPUT_GRAPH = True
        self.SAVED_NETWORK_PATH = './logs/network/'
        self.LOGS_DATA_PATH = './logs/data/'
        self.SAVED_NETWORK_PATH_BACK = './backup/network/'
        self.LOGS_DATA_PATH_BACK = './backup/data/'

        # Class Memory
        self.M_EPSILON = 0.01  # small amount to avoid zero priority
        self.M_ALPHA = 0.6  # [0~1] convert the importance of TD error to priority
        self.M_BETA = 0.4  # importance-sampling, from initial value increasing to 1
        self.M_BETA_INCRE = 0.001
        self.M_ABS_ERROR_UPPER = 1.  # clipped abs error
