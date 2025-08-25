package matrix

import (
	"errors"
	"fmt"
)

// MAKE NOTHING INPLACE
func VecMult(mtx [][]float64, vec []float64) []float64 {
	var result []float64

	// take the vector and rotate it, and pass it through matrix
	for i := 0; i < len(mtx); i++ {
		rowVec := mtx[i]
		sum := 0.0
		for j := 0; j < len(rowVec); j++ {
			sum += rowVec[j] * vec[j]
		}
		result = append(result, sum)
	}

	return result
}

// matrix * matrix
// (m x n) * (n x p) = (m x p)
func MatrixMult(a [][]float64, b [][]float64) ([][]float64, error) {
	var err error
	var result [][]float64

	// do dimension check
	// a has len(a) rows and len(a[0]) columns
	// b has len(b) rows and len(b[0]) columns
	// For multiplication: columns of a must equal rows of b
	if len(a[0]) != len(b) {
		fmt.Println("Dimension doesnt match")
		err = errors.New("dimension doesn't match")
		return result, err
	}

	// append to result, the number of rows there will be
	// which is the the number of rows of a
	for i := 0; i < len(a); i++ {
		result = append(result, []float64{})
	}

	bCols := len(b[0])
	bRows := len(b)
	for i := 0; i < bCols; i++ {
		var colVector []float64
		for j := 0; j < bRows; j++ {
			colVector = append(colVector, b[j][i])
		}

		resColVec := VecMult(a, colVector)

		for i, val := range resColVec {
			result[i] = append(result[i], val)
		}
	}

	return result, err
}

// assumes only 1 row
func RepeatRow(a [][]float64, n int) [][]float64 {
	var res [][]float64

	// := does type inference
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

func RepeatCol(a [][]float64, n int) [][]float64 {
	var result [][]float64

	// First, copy the original row
	firstRow := make([]float64, len(a[0]))
	copy(firstRow, a[0])
	result = append(result, firstRow)

	// Then add n-1 more copies
	for i := 0; i < n-1; i++ {
		rowCopy := make([]float64, len(a[0]))
		copy(rowCopy, a[0])
		result = append(result, rowCopy)
	}

	return result
}

// fill matrix with a value
func FillMatrix(rows int, cols int, val float64) [][]float64 {
	var a [][]float64

	for i := 0; i < rows; i++ {
		a = append(a, []float64{})
		for j := 0; j < cols; j++ {
			a[i] = append(a[i], val)
		}
	}

	return a
}

// elementwise operations - creates new matrix
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

func ScalarMultiply(a [][]float64, scalar float64) [][]float64 {
	// Create a new matrix instead of modifying the original
	var result [][]float64

	for i := 0; i < len(a); i++ {
		result = append(result, []float64{})
		for j := 0; j < len(a[0]); j++ {
			result[i] = append(result[i], a[i][j]*scalar)
		}
	}

	return result
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
	// what is len(a)
	var m int = len(a)
	var n int = len(a[0])

	// Create a copy of the input matrix for ReLU output
	relu := make([][]float64, m)
	for i := range relu {
		relu[i] = make([]float64, n)
	}

	// gradient can start as all 1s and then 0 out
	grad := FillMatrix(m, n, 1)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if a[i][j] < 0 {
				relu[i][j] = 0
				// zero out gradient
				grad[i][j] = 0
			} else {
				relu[i][j] = a[i][j]
			}
		}
	}

	return relu, grad
}
