package nn

import (
	"fmt"
	"nntime/matrix"
)

type Operation interface {
	Perform(mx [][]float64) [][]float64
	UpdateWeight(grad [][]float64, lr float64)
	GetGrad() [][]float64
}

// gradient: alpha function after / alpha this
// for linear, gradient = input
type Linear struct {
	Cols int
	Rows int

	Mtx  [][]float64
	Grad [][]float64
}

// for relu, gradient = 1 or 0
type Relu struct {
	Grad [][]float64
}

// a network is just a list of operations
type Network struct {
	// list of operations
	Ops []Operation
}

func (r *Relu) Perform(mx [][]float64) [][]float64 {
	res, grad := matrix.OpRELU(mx)
	r.Grad = grad
	return res
}

func (r Relu) UpdateWeight(grad [][]float64, lr float64) {
	// no need to update anything
}

func (r Relu) GetGrad() [][]float64 {
	return r.Grad
}

func (l *Linear) Perform(mx [][]float64) [][]float64 {

	l.Grad = mx
	// matrix.RepeatRow(matrix.Transpose(mx), l.Rows)

	var res, _ = matrix.MatrixMult(l.Mtx, mx)

	return res
}

func (l *Linear) UpdateWeight(grad [][]float64, lr float64) {
	// yes, update!

	lrMtx := matrix.FillMatrix(l.Rows, l.Cols, lr)

	l.Mtx = matrix.AddE(l.Mtx, matrix.MultE(lrMtx, grad))
}

func (l Linear) GetGrad() [][]float64 {
	return l.Grad
}

func (n Network) Perform(mx [][]float64) [][]float64 {

	for _, op := range n.Ops {
		mx = op.Perform(mx)
	}

	return mx
}

// update weights
func (n Network) Update(lr float64) {
	var gradUpdate [][]float64

	// go backwards
	for o := len(n.Ops); o > 0; o-- {
		// o - 1
		op := n.Ops[o-1]

		// last operation
		if o == len(n.Ops) {
			gradUpdate = op.GetGrad()
		} else {
			// gradUpdate * transpose of new op's gradient
			gradUpdate, _ = matrix.MatrixMult(gradUpdate, matrix.Transpose(op.GetGrad()))
			fmt.Println(gradUpdate)

		}
		// if weights layer actually update it
		op.UpdateWeight(gradUpdate, lr)
	}

}
