
from src.configs.interface import DefaultInterface
from src.configs.cfg import AutoDict



class SimpleClf(DefaultInterface):

    main_tag        = 'Classifier'
    metrics         = [ AutoDict(name=fn) for fn in 'accuracy_score,f1_score,roc'.split(',') ]