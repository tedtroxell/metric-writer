
import abc

class DefaultInterface(metaclass=abc.ABCMeta):
    """
        Listen, I know that the class name says "Interface" and then I'm making Abstract Class, but it's Python, give me a break.
    """
    
    def __init__(self,*args,**kwargs):
        super( DefaultInterface,self ).__init__(*args,**kwargs)
        print(args,kwargs)

    @staticmethod
    @abc.abstractmethod
    def auto_config(cls:'DefaultInterface', model:object):
        """
            Autogenerate a default config from an arbitrary model.
        """
        pass

    