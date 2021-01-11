
from typing import Union
from src.configs.cfg import DefaultCFG
class MetricWriter:
    """
        Metric Writer is a lightweight class that will automatically log metrics during a training loop.
    """

    def __init__(   self,
                    cfg:Union[dict,DefaultCFG] = None
                )-> 'MetricWriter': 
        from torch.utils.tensorboard import SummaryWriter
        import sklearn.metrics as metrics
        from src.configs.interface import DefaultInterface

        # check for user provided config, 
        # if not present fallback to default
        self.cfg = DefaultCFG( cfg if isinstance( cfg, dict ) else None ) if not issubclass(cfg,DefaultInterface) else cfg
        self.writer = SummaryWriter()
        self.metrics = metrics
        self.index = 0

    def close(self):
        self.writer.close()


    def __call__(   self,
                    _output : object, 
                    _labels: object, 
                    loss_fn: object = None,
                    index : int = -1
                ) -> Union[ 'loss','self' ]:

        """

        """
        self.index += 1
        output = _output.clone().detach().argmax(dim=-1).numpy()
        labels = _labels.clone().detach().argmax(dim=-1).numpy()
        for metric in self.cfg.metrics:
            if hasattr( self.metrics,metric.name ): self.writer.add_scalar(
                self.cfg.main_tag+'_'+metric.name,
                getattr(self.metrics,metric.name)( output,labels ),
            index if index > -1 else self.index
        )
        if callable(loss_fn):
            loss = loss_fn( _output,_labels )
            self.writer.add_scalar(self.cfg.main_tag+'_loss',loss.item(),index if index > -1 else self.index)
            return loss
        return self
