package nn

import (
	"fmt"
	"nntime/matrix"
)

type Operation interface {
	Perform(mx [][]float64) [][]float64
	UpdateWeight(grad [][]float64, lr float64)

	// first we compute grad to be
	// gradient of the next layer's neurons over this neuron

	// then we work backwards to mean aloss/ainput
	GetGrad() [][]float64
	SetGrad([][]float64)

	isActivation() bool
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

type Loss struct {
	Grad [][]float64
}

// a network is just a list of operations
type Network struct {
	// list of operations
	Ops []Operation
}

// performing our loss layer
func (l *Loss) Perform(mx [][]float64) [][]float64 {
	// return 1 number as matrix
	var lossMtx [][]float64 = [][]float64{{0}}
	var grad [][]float64

	vec := mx[0]

	for i := 0; i < len(vec); i++ {
		// for now -> our loss
		lossMtx[0][0] += vec[i] * vec[i]

		// append to l.Grad 2 * vec[i]
		grad = append(grad, []float64{2 * vec[i]})
	}

	l.Grad = grad
	return lossMtx
}

func (l Loss) UpdateWeight(grad [][]float64, lr float64) {
	// no need to update anything
}

func (l Loss) GetGrad() [][]float64 {
	return l.Grad
}

func (l *Loss) SetGrad(grad [][]float64) {
	l.Grad = grad
}

func (l Loss) isActivation() bool {
	return false
}

func (r *Relu) Perform(mx [][]float64) [][]float64 {
	res, grad := matrix.OpRELU(mx)
	r.Grad = grad
	return res
}

func (r Relu) UpdateWeight(grad [][]float64, lr float64) {
	// no need to update anything
}

func (r *Relu) SetGrad(grad [][]float64) {
	r.Grad = grad
}

func (r Relu) GetGrad() [][]float64 {
	return r.Grad
}

func (r Relu) isActivation() bool {
	return true
}

func (l *Linear) Perform(mx [][]float64) [][]float64 {

	// gradient is weights
	l.Grad = l.Mtx
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

func (l *Linear) SetGrad(grad [][]float64) {
	l.Grad = grad
}

func (l Linear) isActivation() bool {
	return false
}

func (n Network) Perform(mx [][]float64) [][]float64 {

	for _, op := range n.Ops {
		mx = op.Perform(mx)
	}

	return mx
}

// update weights
// assumes always linear before activation
func (n Network) Update(lr float64) {
	// first step is to go backwards
	// and collect aloss/ainput gradients
	o := len(n.Ops)
	gradUpdate := n.Ops[o-1].GetGrad()
	o -= 1
	condition := true

	for condition {
		fmt.Println(o)

		if o < 1 {
			break
		}

		// first check if isActivation
		newOp := n.Ops[o-1]
		var newGrad [][]float64

		newGrad = newOp.GetGrad()

		// but if activation
		if newOp.isActivation() {
			temp := newOp.GetGrad()

			o -= 1
			newOp = n.Ops[o-1]

			// elementwise multiplication
			// with 0's and 1's in ReLU
			newGrad = matrix.MultE(temp, newOp.GetGrad())
		}

		// then carry on how you were before

		// [[6]]
		fmt.Println(gradUpdate)
		// [[0 0 1]]

		// WE FORGOT ABOUT THE 2 inputs (ouch)
		fmt.Println(newGrad)
		gradUpdate, _ = matrix.MatrixMult(gradUpdate, newGrad)
		newOp.SetGrad(gradUpdate)

		o -= 1
	}

	// second step is to go and actually update the weights

}
