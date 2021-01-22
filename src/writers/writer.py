
from typing import Union
from metric_writer.src.configs.cfg import DefaultCFG
from torch import Tensor
class MetricWriter:
    """
        Metric Writer is a lightweight class that will automatically log metrics during a training loop.
    """

    def __init__(   self,
                    cfg:Union[dict,DefaultCFG] = None
                )-> 'MetricWriter': 
        from torch.utils.tensorboard import SummaryWriter
        import sklearn.metrics as metrics
        from metric_writer.src.configs.interface import DefaultConfigInterface

        # check for user provided config, 
        # if not present fallback to default
        self.cfg = DefaultCFG( cfg if isinstance( cfg, dict ) else None ) if not issubclass(cfg.__class__,DefaultConfigInterface) else cfg
        self.writer = SummaryWriter()
        self.metrics = metrics
        self.custom_functions = {}
        self.index = 0

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

    @staticmethod
    def from_model(model):
        from metric_writer.src.configs.interface import DefaultConfigInterface
        return MetricWriter( DefaultConfigInterface.auto_config( model ) )

    def close(self):self.writer.close()

    def __exit__(self):self.close()

    def __call__(   self,
                    _output : Tensor, 
                    _labels: Tensor, 
                    loss_fn: 'PyTorch Loss Function' = None,
                    index : int = -1
                ) -> Union[ 'loss','self' ]:

        """

        """
        self.index += 1
        output,labels = self.cfg.sanitize_inputs( _output, _labels )
        for metric in self.cfg.metrics:
            if hasattr( self.metrics,metric.name ): self.writer.add_scalar(
                self.cfg.main_tag+'_'+metric.name,
                getattr(self.metrics,metric.name)( output,labels ),
            index if index > -1 else self.index
        )
        for name,func in self.custom_functions.items():
            self.writer.add_scalar(
                                    self.cfg.main_tag+'_'+name,
                                    func( output,labels ),
                                    index if index > -1 else self.index
            )
        if callable(loss_fn):
            loss = loss_fn( _output,_labels )
            self.writer.add_scalar(self.cfg.main_tag+'_loss',loss.item(),index if index > -1 else self.index)
            return loss
        return self
