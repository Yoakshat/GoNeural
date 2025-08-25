package main

import (
	"nntime/nn"
)

func main() {
	// weight * input (Wx)
	// 4 rows x 1 column
	input := [][]float64{{1}, {2}, {-1}, {0}}
	// 3 rows x 4 columns
	w1 := [][]float64{{1, 2, -1, 0}, {0, -1, 5, 1}, {4, -2, 1, 0}}
	// 2 rows x 3 columns
	w2 := [][]float64{{3, 1, 2}, {2, 0, 1}}
	labelProbs := [][]float64{{0.5}, {0.5}}

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
}
