
from typing import Union

class MetricWriter:
    """
        Metric Writer is a lightweight class that will automatically log metrics during a training loop.
    """

    def __init__(   self,
                    cfg:dict = None
                )-> 'MetricWriter': 
        # from metric_writer import DefaultCFG
        from torch.utils.tensorboard import SummaryWriter
        import sklearn.metrics as metrics

        # check for user provided config, 
        # if not present fallback to default
        self.cfg = DefaultCFG( cfg if isinstance( cfg, (None,dict) ) else None ) 
        self.writer = SummaryWriter()
        self.metrics = metrics
        self.index = 0


    def __call__(   self,
                    output : object, 
                    labels: object, 
                    loss_fn: object = None,
                    index : int = -1
                ) -> Union[ loss,self ]:

        """

        """
        self.index += 1
        self.writer.add_custom_scalars(
            self.cfg.main_tag,
            {
                metric.name:getattr(self.metrics,metric.name)( output,labels ) for metric in self.cfg.metrics if hasattr( self.metrics,metric.name )
            },
            index if index > -1 else self.index
        )
        if callable(loss_fn):
            loss = loss_fn( output,labels )
            self.writer.add_scalar(self.cfg.main_tag+'/loss',loss.item(),index if index > -1 else self.index)
            return loss
        return self
