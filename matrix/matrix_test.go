package matrix

import (
	"math"
	"testing"
)

// Helper function to compare float slices with tolerance
func floatSliceEqual(a, b []float64, tolerance float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tolerance {
			return false
		}
	}
	return true
}

// Helper function to compare float matrices with tolerance
func matrixEqual(a, b [][]float64, tolerance float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !floatSliceEqual(a[i], b[i], tolerance) {
			return false
		}
	}
	return true
}

func TestVecMult(t *testing.T) {
	// Test case 1: 2x2 matrix * 2-element vector
	mtx1 := [][]float64{
		{1, 2},
		{3, 4},
	}
	vec1 := []float64{2, 1}
	expected1 := []float64{5, 11} // [1*2+3*1, 2*2+4*1]
	result1 := VecMult(mtx1, vec1)

	if !floatSliceEqual(result1, expected1, 1e-9) {
		t.Errorf("VecMult test 1 failed: expected %v, got %v", expected1, result1)
	}

	// Test case 2: 3x2 matrix * 3-element vector
	// It is how it looks (3 rows and 2 columns)
	mtx2 := [][]float64{
		{1, 0},
		{0, 1},
		{2, 3},
	}
	vec2 := []float64{4, 5, 6}
	expected2 := []float64{16, 23} // [1*4+0*5+2*6, 0*4+1*5+3*6]
	result2 := VecMult(mtx2, vec2)

	if !floatSliceEqual(result2, expected2, 1e-9) {
		t.Errorf("VecMult test 2 failed: expected %v, got %v", expected2, result2)
	}
}

func TestMatrixMult(t *testing.T) {
	// Test case 1: Valid multiplication 2x2 * 2x2
	a1 := [][]float64{
		{1, 2},
		{3, 4},
	}
	b1 := [][]float64{
		{5, 6},
		{7, 8},
	}
	expected1 := [][]float64{
		{19, 22}, // [1*5+2*7, 1*6+2*8]
		{43, 50}, // [3*5+4*7, 3*6+4*8]
	}
	result1, err1 := MatrixMult(a1, b1)

	if err1 != nil {
		t.Errorf("MatrixMult test 1 failed with error: %v", err1)
	}
	if !matrixEqual(result1, expected1, 1e-9) {
		t.Errorf("MatrixMult test 1 failed: expected %v, got %v", expected1, result1)
	}

	// Test case 2: Invalid dimensions
	a2 := [][]float64{
		{1, 2, 3},
	}
	b2 := [][]float64{
		{1, 2},
		{3, 4},
	}
	_, err2 := MatrixMult(a2, b2)

	if err2 == nil {
		t.Errorf("MatrixMult test 2 should have failed with dimension mismatch")
	}

	// Test case 3: 2x3 * 3x1
	a3 := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}
	b3 := [][]float64{
		{1},
		{2},
		{3},
	}
	expected3 := [][]float64{
		{14}, // 1*1+2*2+3*3
		{32}, // 4*1+5*2+6*3
	}
	result3, err3 := MatrixMult(a3, b3)

	if err3 != nil {
		t.Errorf("MatrixMult test 3 failed with error: %v", err3)
	}
	if !matrixEqual(result3, expected3, 1e-9) {
		t.Errorf("MatrixMult test 3 failed: expected %v, got %v", expected3, result3)
	}
}

func TestRepeatRow(t *testing.T) {
	// Test case 1: Repeat single element 3 times
	a1 := [][]float64{
		{5},
		{7},
	}
	expected1 := [][]float64{
		{5, 5, 5},
		{7, 7, 7},
	}
	result1 := RepeatRow(a1, 3)

	if !matrixEqual(result1, expected1, 1e-9) {
		t.Errorf("RepeatRow test 1 failed: expected %v, got %v", expected1, result1)
	}

	// Test case 2: Single row
	a2 := [][]float64{
		{2},
	}
	expected2 := [][]float64{
		{2, 2, 2, 2},
	}
	result2 := RepeatRow(a2, 4)

	if !matrixEqual(result2, expected2, 1e-9) {
		t.Errorf("RepeatRow test 2 failed: expected %v, got %v", expected2, result2)
	}
}

func TestRepeatCol(t *testing.T) {
	// Test case 1: Repeat column 3 times
	a1 := [][]float64{
		{1, 2},
	}
	expected1 := [][]float64{
		{1, 2},
		{1, 2},
		{1, 2},
	}
	result1 := RepeatCol(a1, 3)

	if !matrixEqual(result1, expected1, 1e-9) {
		t.Errorf("RepeatCol test 1 failed: expected %v, got %v", expected1, result1)
	}

	// Test case 2: Single element column
	a2 := [][]float64{
		{5},
	}
	expected2 := [][]float64{
		{5},
		{5},
	}
	result2 := RepeatCol(a2, 2)

	if !matrixEqual(result2, expected2, 1e-9) {
		t.Errorf("RepeatCol test 2 failed: expected %v, got %v", expected2, result2)
	}
}

func TestFillMatrix(t *testing.T) {
	// Test case 1: 3x2 matrix filled with 7.5
	result1 := FillMatrix(3, 2, 7.5)
	expected1 := [][]float64{
		{7.5, 7.5},
		{7.5, 7.5},
		{7.5, 7.5},
	}

	if !matrixEqual(result1, expected1, 1e-9) {
		t.Errorf("FillMatrix test 1 failed: expected %v, got %v", expected1, result1)
	}

	// Test case 2: 1x1 matrix
	result2 := FillMatrix(1, 1, -3.14)
	expected2 := [][]float64{
		{-3.14},
	}

	if !matrixEqual(result2, expected2, 1e-9) {
		t.Errorf("FillMatrix test 2 failed: expected %v, got %v", expected2, result2)
	}
}

func TestElementwise(t *testing.T) {
	a := [][]float64{
		{1, 2},
		{3, 4},
	}
	b := [][]float64{
		{5, 6},
		{7, 8},
	}

	// Test addition
	expectedAdd := [][]float64{
		{6, 8},
		{10, 12},
	}
	resultAdd := Elementwise(a, b, add)

	if !matrixEqual(resultAdd, expectedAdd, 1e-9) {
		t.Errorf("Elementwise add failed: expected %v, got %v", expectedAdd, resultAdd)
	}

	// Test multiplication
	expectedMult := [][]float64{
		{5, 12},
		{21, 32},
	}
	resultMult := Elementwise(a, b, multiply)

	if !matrixEqual(resultMult, expectedMult, 1e-9) {
		t.Errorf("Elementwise multiply failed: expected %v, got %v", expectedMult, resultMult)
	}
}

func TestAddE(t *testing.T) {
	a := [][]float64{
		{1.5, 2.5},
		{3.5, 4.5},
	}
	b := [][]float64{
		{0.5, 1.5},
		{2.5, 3.5},
	}
	expected := [][]float64{
		{2.0, 4.0},
		{6.0, 8.0},
	}
	result := AddE(a, b)

	if !matrixEqual(result, expected, 1e-9) {
		t.Errorf("AddE failed: expected %v, got %v", expected, result)
	}
}

func TestMultE(t *testing.T) {
	a := [][]float64{
		{2, 3},
		{4, 5},
	}
	b := [][]float64{
		{1, 2},
		{3, 4},
	}
	expected := [][]float64{
		{2, 6},
		{12, 20},
	}
	result := MultE(a, b)

	if !matrixEqual(result, expected, 1e-9) {
		t.Errorf("MultE failed: expected %v, got %v", expected, result)
	}
}

func TestScalarMultiply(t *testing.T) {
	// Note: This function modifies the input matrix
	a := [][]float64{
		{1, 2},
		{3, 4},
	}
	expected := [][]float64{
		{2.5, 5.0},
		{7.5, 10.0},
	}
	result := ScalarMultiply(a, 2.5)

	if !matrixEqual(result, expected, 1e-9) {
		t.Errorf("ScalarMultiply failed: expected %v, got %v", expected, result)
	}
}

func TestTranspose(t *testing.T) {
	// Test case 1: 2x3 matrix
	a1 := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}
	expected1 := [][]float64{
		{1, 4},
		{2, 5},
		{3, 6},
	}
	result1 := Transpose(a1)

	if !matrixEqual(result1, expected1, 1e-9) {
		t.Errorf("Transpose test 1 failed: expected %v, got %v", expected1, result1)
	}

	// Test case 2: Square matrix
	a2 := [][]float64{
		{1, 2},
		{3, 4},
	}
	expected2 := [][]float64{
		{1, 3},
		{2, 4},
	}
	result2 := Transpose(a2)

	if !matrixEqual(result2, expected2, 1e-9) {
		t.Errorf("Transpose test 2 failed: expected %v, got %v", expected2, result2)
	}

	// Test case 3: Single row
	a3 := [][]float64{
		{1, 2, 3, 4},
	}
	expected3 := [][]float64{
		{1},
		{2},
		{3},
		{4},
	}
	result3 := Transpose(a3)

	if !matrixEqual(result3, expected3, 1e-9) {
		t.Errorf("Transpose test 3 failed: expected %v, got %v", expected3, result3)
	}
}

func TestOpRELU(t *testing.T) {
	// Test with mixed positive and negative values
	a := [][]float64{
		{-1, 2, -3},
		{4, -5, 6},
	}
	expectedRelu := [][]float64{
		{0, 2, 0},
		{4, 0, 6},
	}
	expectedGrad := [][]float64{
		{0, 1, 0},
		{1, 0, 1},
	}

	resultRelu, resultGrad := OpRELU(a)

	if !matrixEqual(resultRelu, expectedRelu, 1e-9) {
		t.Errorf("OpRELU ReLU failed: expected %v, got %v", expectedRelu, resultRelu)
	}
	if !matrixEqual(resultGrad, expectedGrad, 1e-9) {
		t.Errorf("OpRELU gradient failed: expected %v, got %v", expectedGrad, resultGrad)
	}

	// Test with all positive values
	a2 := [][]float64{
		{1, 2},
		{3, 4},
	}
	expectedRelu2 := [][]float64{
		{1, 2},
		{3, 4},
	}
	expectedGrad2 := [][]float64{
		{1, 1},
		{1, 1},
	}

	resultRelu2, resultGrad2 := OpRELU(a2)

	if !matrixEqual(resultRelu2, expectedRelu2, 1e-9) {
		t.Errorf("OpRELU all positive ReLU failed: expected %v, got %v", expectedRelu2, resultRelu2)
	}
	if !matrixEqual(resultGrad2, expectedGrad2, 1e-9) {
		t.Errorf("OpRELU all positive gradient failed: expected %v, got %v", expectedGrad2, resultGrad2)
	}

	// Test with all negative values
	a3 := [][]float64{
		{-1, -2},
		{-3, -4},
	}
	expectedRelu3 := [][]float64{
		{0, 0},
		{0, 0},
	}
	expectedGrad3 := [][]float64{
		{0, 0},
		{0, 0},
	}

	resultRelu3, resultGrad3 := OpRELU(a3)

	if !matrixEqual(resultRelu3, expectedRelu3, 1e-9) {
		t.Errorf("OpRELU all negative ReLU failed: expected %v, got %v", expectedRelu3, resultRelu3)
	}
	if !matrixEqual(resultGrad3, expectedGrad3, 1e-9) {
		t.Errorf("OpRELU all negative gradient failed: expected %v, got %v", expectedGrad3, resultGrad3)
	}
}
