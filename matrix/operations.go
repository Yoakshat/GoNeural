package matrix

import (
	"errors"
	"fmt"
)

// NOTHING IS INPLACE

// matrix * vector
func VecMult(mtx [][]float64, vec []float64) []float64 {
	// check if dimension equal
	var endVec []float64
	var mtxDim int = len(mtx[0])

	// multiply vector into matrix
	for comp := 0; comp < mtxDim; comp++ {
		var combo float64 = 0
		for i := 0; i < len(vec); i++ {
			combo += vec[i] * mtx[i][comp]
		}

		endVec = append(endVec, combo)
	}

	return endVec

}

// matrix * matrix
func MatrixMult(a [][]float64, b [][]float64) ([][]float64, error) {
	var err error
	var result [][]float64

	// check match dimensions here, instead
	if len(a) != len(b[0]) {
		fmt.Print("Dimension doesnt match")
		err = errors.New("dimension doesn't match")
		return result, err
	}

	for idxVec := 0; idxVec < len(b); idxVec++ {
		result = append(result, VecMult(a, b[idxVec]))
	}

	return result, err
}

// assumes only 1 row
func RepeatRow(a [][]float64, n int) [][]float64 {
	var res [][]float64

	for vec := 0; vec < len(a); vec++ {
		// what we want to repeat
		var element float64 = a[vec][0]
		res = append(res, []float64{})
		// already there once
		for i := 0; i < n; i++ {
			res[vec] = append(res[vec], element)
		}
	}

	return res
}

// fill matrix with a value
func FillMatrix(rows int, cols int, val float64) [][]float64 {

	var a [][]float64

	for i := 0; i < cols; i++ {
		a = append(a, []float64{})
		for j := 0; j < rows; j++ {
			a[i] = append(a[i], val)
		}
	}

	return a
}

// elementwise is inplace
func Elementwise(a [][]float64, b [][]float64, oper func(float64, float64) float64) [][]float64 {
	var res [][]float64

	for i := 0; i < len(a); i++ {
		res = append(res, []float64{})
		for j := 0; j < len(a[0]); j++ {
			res[i] = append(res[i], oper(a[i][j], b[i][j]))
		}
	}

	return res
}

func AddE(a [][]float64, b [][]float64) [][]float64 {
	return Elementwise(a, b, add)
}

func MultE(a [][]float64, b [][]float64) [][]float64 {
	return Elementwise(a, b, multiply)
}

func multiply(a float64, b float64) float64 {
	return a * b
}

func add(a float64, b float64) float64 {
	return a + b
}

func Transpose(a [][]float64) [][]float64 {
	// switch the rows and columns
	var a_t [][]float64

	for i := 0; i < len(a[0]); i++ {
		// add empty array
		a_t = append(a_t, []float64{})
		for j := 0; j < len(a); j++ {
			// add to this empty array
			a_t[i] = append(a_t[i], a[j][i])
		}
	}

	return a_t
}

// returns actual ReLU, and gradients
func OpRELU(a [][]float64) ([][]float64, [][]float64) {
	var m int = len(a)
	var n int = len(a[0])

	var gradMtx [][]float64

	for i := 0; i < m; i++ {
		gradMtx = append(gradMtx, []float64{})
		for j := 0; j < n; j++ {
			if a[i][j] < 0 {
				a[i][j] = 0
				gradMtx[i] = append(gradMtx[i], 0)
			} else {
				gradMtx[i] = append(gradMtx[i], 1)
			}
		}
	}

	return a, gradMtx
}
