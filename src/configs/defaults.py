
from src.configs.interface import DefaultInterface
from src.configs.cfg import AutoDict



class SimpleClassification(DefaultInterface):

    main_tag        = 'Classifier'
    metrics         = [ AutoDict(name=fn) for fn in 'accuracy,f1,roc_auc'.split(',') ]