package main

import (
	"fmt"
	"nntime/nn"
)

func main() {
	input := [][]float64{{1, 2, -1, 0}}
	w1 := [][]float64{{1, 0, 1}, {0, -1, 0},
		{1, -1, 0}, {1, 1, 1}}
	// 3 columns, 2 rows
	w2 := [][]float64{{3, 1}, {2, 0}, {1, 2}}

	// in theory, as many linears and relus in fashion should work
	// to try out tmrw!

	layer := []nn.Operation{
		&nn.Linear{Cols: 4, Rows: 3, Mtx: w1},
		&nn.Relu{},
		&nn.Linear{Cols: 3, Rows: 2, Mtx: w2},
		&nn.Loss{},
	}
	feedforward := nn.Network{Ops: layer}
	res := feedforward.Perform(input)

	fmt.Println(res)

	// facing errors with our updaate
	feedforward.Update(1e-3)

	// fmt.Println(input)

}
