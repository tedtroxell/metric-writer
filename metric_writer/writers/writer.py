
from typing import Union,Optional
from metric_writer.configs.cfg import DefaultCFG
from torch import Tensor
from metric_writer.writers.interface import BaseWriter

MetricWriter = None
class MetricWriter(BaseWriter):
    """
        Metric Writer is a lightweight class that will automatically log metrics during a training loop.
    """

    def __init__(   self,
                    cfg:Union[dict,DefaultCFG] = None
                )-> MetricWriter: 
        super(self.__class__,self).__init__()
        
        import sklearn.metrics as metrics
        from metric_writer.configs.interface import DefaultConfigInterface

        # check for user provided config, 
        # if not present fallback to default
        self.cfg = DefaultCFG( cfg if isinstance( cfg, dict ) else None ) if not issubclass(cfg.__class__,DefaultConfigInterface) else cfg
        
        self.metrics = metrics
        self.custom_functions = {}
        self.signals = {}
        self.montiors = {}
        self.index = 0

    def forward(   self,
                    _output : Tensor, 
                    _labels: Tensor, 
                    loss_fn: Optional[callable] = None,
                    index : Optional[int] = -1
                ) -> Union[ 'loss',MetricWriter ]:

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
