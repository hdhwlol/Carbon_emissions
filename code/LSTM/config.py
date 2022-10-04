class Config():

    def __init__(self
                 ,input_size
                 ,hidden_size
                 ,num_layers
                 ,output_size
                 ,bidirectional
                 ,optimizer
                 ,epochs
                 ,batch_size
                 ,lr
                 ,weight_decay):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        # self.step_size = step_size
        # self.gamma = gamma