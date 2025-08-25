package nn

import (
	"nntime/matrix"
)

type Operation interface {
	Forward(mx [][]float64) [][]float64
	Backward(mx [][]float64) [][]float64
	SetGrad(grad [][]float64)
}

type LossLayer interface {
	Forward(prediction [][]float64, target [][]float64) float64
	GetGrad() [][]float64
}

// gradient: alpha function after / alpha this
// for linear, gradient = input
type Linear struct {
	Cols int
	Rows int

	Mtx        [][]float64
	Grad       [][]float64
	WeightGrad [][]float64
}

type Relu struct {
	Grad [][]float64
}

type MSELoss struct {
	Grad [][]float64
}

// a network is just a list of operations
// also should have a list of linear layers
type Network struct {
	// list of operations
	Ops     []Operation
	Linears []*Linear
	Loss    LossLayer
}

func NeuralNetwork(Ops []Operation, Loss LossLayer) *Network {
	linears := []*Linear{}

	for _, op := range Ops {
		l, ok := op.(*Linear)
		if ok {
			linears = append(linears, l)
		}
	}

	return &Network{Ops: Ops, Linears: linears, Loss: Loss}
}

// performing our loss layer
func (l *MSELoss) Forward(yhat [][]float64, y [][]float64) float64 {
	// return 1 number as matrix
	loss := 0.0

	for i := 0; i < len(yhat); i++ {
		pred := yhat[i][0]
		targ := y[i][0]

		loss += (pred - targ) * (pred - targ)
		l.Grad = append(l.Grad, []float64{2 * (pred + targ)})
	}

	// loss over number of samples (only 1 sample for now)
	return loss
}

func (l *MSELoss) GetGrad() [][]float64 {
	return l.Grad
}

func (r *Relu) Forward(mx [][]float64) [][]float64 {
	res, grad := matrix.OpRELU(mx)
	r.Grad = grad
	return res
}

func (r *Relu) Backward(mx [][]float64) [][]float64 {
	// series of 0s and 1s
	// elementwise multiply of mx and 1's and 0's
	res := matrix.MultE(mx, r.Grad)
	return res
}

func (r *Relu) SetGrad(grad [][]float64) {
	r.Grad = grad
}

func (l *Linear) Forward(mx [][]float64) [][]float64 {
	// get that dimension doesn't match
	res, _ := matrix.MatrixMult(l.Mtx, mx)
	// alpha next input / last input are just weights
	l.Grad = matrix.Transpose(l.Mtx)
	// repeat column for the number of rows res has
	l.WeightGrad = matrix.RepeatCol(mx, len(res))
	return res
}

func (l *Linear) Backward(mx [][]float64) [][]float64 {
	res, _ := matrix.MatrixMult(l.Grad, mx)
	return res
}

func (l *Linear) SetGrad(grad [][]float64) {
	l.Grad = grad
}

// perform with labels
func (n Network) Forward(mx [][]float64, target [][]float64) float64 {
	for _, op := range n.Ops {
		mx = op.Forward(mx)
	}

	// perform loss layer
	return n.Loss.Forward(mx, target)
}

// idea is to turn local gradients into global gradients
func (n Network) Backward() {
	res := n.Loss.GetGrad()
	// go backwards through the network
	for i := len(n.Ops) - 1; i >= 0; i-- {
		// turning into global gradient
		// every operation has a gradient, so we need a SetGrad
		res = n.Ops[i].Backward(res)
		n.Ops[i].SetGrad(res)
	}

	// multiply the (alpha input / alpha weight gradients) * (alpha Loss/alpha input)
	// intuition is (gradients of layer that created those inputs) * (future loss/input gradients)
	for i := len(n.Linears) - 1; i >= 0; i-- {
		lossInputGrad := n.Loss.GetGrad()
		if i < len(n.Linears)-1 {
			lossInputGrad = n.Linears[i+1].Grad
		}

		localWeightGrad := n.Linears[i].WeightGrad
		// turning into global gradient
		// repeat for the number of rows the local weight gradient has
		n.Linears[i].WeightGrad = matrix.MultE(localWeightGrad, matrix.RepeatRow(lossInputGrad, len(localWeightGrad)))
	}
}

func (n Network) Update(lr float64) {
	// only linear layers have weights
	for _, lin := range n.Linears {
		// update weights based on global weight gradients with respect to loss
		// weights += lr * alpha loss / alpha weight
		lin.Mtx = matrix.AddE(lin.Mtx, matrix.ScalarMultiply(lin.WeightGrad, lr))
	}
}
