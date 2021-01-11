# Metric Writer

Metric Writer is a small lightweight module that easily and automatically records performance metrics for you. 

## Usage
```python
import torch
from torch import nn
from metric_writer import MetricWriter, Defaults

# initialize data
X = torch.randn( 10, 10 ) # input size 10
Y = torch.randn( 10,4 ) # 4 classes

# initialize simple model, loss and optimizer
model = nn.Linear(10,4)
optim = torch.optim.SGD(model.parameters())
loss_fn = nn.L1Loss()

# initialize Metric Writer
mw = MetricWriter( Defaults.SimpleClf ) # initialize with basic classifier metrics

# inside your training loop
for _ in range(EPOCHS):
	y = model( X )
	# you can either pass the loss function into the metric writer to automatically
	# record the loss or do that seperately. If you pass in the loss, it will automatically
	# call your loss function for you i.e. (loss = loss_fn(yhat,y) ) and return the result
	# if you are recording every couple of steps/batches it's recommended that you do not pass
	# the loss function to the metric writer
	loss = mw(y,Y,loss_fn)
	loss.backward()
	optim.step()
```

## TODOs
* add histograms, embeddings and graphs
* autoconfig from model
* ?????
* profit