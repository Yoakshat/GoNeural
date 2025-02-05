package main

import (
	"fmt"
	"nntime/nn"
)

func main() {
	input := [][]float64{{1, 2, -1, 0}}
	w1 := [][]float64{{1, 0, 1}, {0, -1, 0},
		{1, -1, 0}, {1, 1, 1}}

	layer := []nn.Operation{&nn.Linear{Cols: 4, Rows: 3, Mtx: w1}, &nn.Relu{}}
	feedforward := nn.Network{Ops: layer}
	res := feedforward.Perform(input)

	fmt.Println(res)

	feedforward.Update(1e-3)

	// fmt.Println(input)

}
