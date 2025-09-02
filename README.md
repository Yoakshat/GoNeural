A neural network from scratch! 
- Implements linear layers, activation functions like ReLU, and matrix operations all from scratch. 
- We also implement backpropagation from scratch (using the same backward/forward ideas of PyTorch). Partial gradients are computed from scratch.

This is an extensible framework. You can stack linears, ReLUs. You can also create new layers, but you have to define their Forwards and Backwards. 
Here is an example. You can initialize your weights however you wish: 

```
layer := []nn.Operation{
		&nn.Linear{Cols: 4, Rows: 3, Mtx: w1},
		&nn.Relu{},
		&nn.Linear{Cols: 3, Rows: 2, Mtx: w2},
	}
loss := &nn.MSELoss{}
feedforward := nn.Network{Ops: layer, Loss: loss}
// then compute loss
feedforward.Forward(input, labelProbs)
feedforward.Backward()
feedforward.Update(1e-3)
```

In Progress
- handle multi-batches
- enable parallel computation on GPUs
- train on simple task like MNIST

