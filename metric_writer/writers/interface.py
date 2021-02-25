
from metric_writer.interface import BaseInterface
BaseWriter = None
class BaseWriter(BaseInterface):

    montitors = {}

    def __init__(self,*args,**kwargs):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter()

    def register_monitor(self, kls_fn : callable ) -> BaseWriter:
        self.montitors[kls_fn.__class__.__name__] = kls_fn
        return self

    
    def unregister_monitor(self, kls_fn : callable ) -> BaseWriter:
        del self.montitors[kls_fn.__class__.__name__]
        return self

    def register_metric(self,
                        name: str, 
                        func: callable
                        ) -> 'self':
        """
            Add a callable function that will be used to evaluate the RAW data.
            If you need to coerce the data in any way, you will need to handle that on your own.

            Args:
                name: The name of the metric you want to log. This will be recorded in tensorboard
                func: The callable function that will consume the RAW data. ``i.e. (yhat,y)``
        """
        self.custom_functions[name] = func
        return self

    @staticmethod
    def from_model(model) -> BaseWriter:
        from metric_writer.configs.interface import DefaultConfigInterface
        return BaseWriter( DefaultConfigInterface.auto_config( model ) )

    def close(self):self.writer.close()

    def __exit__(self):self.close()
