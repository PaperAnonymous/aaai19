import numpy as np

class Config(object):
    """
    Configuration for bAbI
    """
    def __init__(self, train_story, train_questions, train_qstory, dictionary, outfile):
        self.batch_size       = 32
        self.nhops            = 1
        self.nepochs          = 10
        self.lrate_decay_step = 25   # reduce learning rate by half every 25 epochs
        self.outfile = outfile
        self.init_vocab = True

        if isinstance(train_story, np.ndarray):
            train_story = train_story.shape[-1]

        # Training configuration
        self.train_config = {
            "init_lrate"   : 0.01,
            "max_grad_norm": 500,
            "in_dim"       : 100,
            "dropout"      : 0.9,
            "sz"           : min(10, train_story),
            "voc_sz"       : len(dictionary),
            "bsz"          : self.batch_size,
            "max_words"    : 10,
            "weight"       : None,
            "emb_query_weight": None,
            "emb_out_weight": None,
            "candidates_num": 1,
        }

