// reference sheet to look back
// learning go

package nn

import (
	"errors"
	"fmt"
)

type library struct {
	numBooks   int
	holdBooks  int
	squareFeet int
}

type bookPlace interface {
	booksLeft() int
}

// like functions in a class
// there is no name
func (b library) booksLeft() int {
	return (b.numBooks - b.holdBooks)
}

func (b library) multiply() int {
	return b.squareFeet * b.numBooks
}

// make functions capital
// learning syntax
func Neural(printMe int) (int, error) {
	var err error
	if printMe == 0 {
		err = errors.New("cannot do this")
		return printMe, err
	}

	fmt.Println(printMe)

	var n int = 0
	for i := 0; i < 400; i++ {
		n += 1
	}

	fmt.Println(n)

	// shorthand for = without var
	myVar := "text"
	fmt.Println(myVar)

	return n, err
}

func ArrayStuff() {
	intArr := [3]int32{1, 2, 3}
	fmt.Println(intArr)

	var intSlice []int = []int{4, 5, 6}
	intSlice = append(intSlice, 7)
	fmt.Printf("%v", len(intSlice))

	var myMap map[string]int = map[string]int{"Adam": 5, "V": 3}
	fmt.Println(myMap)

}

func StructStuff() {
	var myLibrary library = library{10, 5, 1500}
	fmt.Println(myLibrary.numBooks)
	fmt.Println(myLibrary.booksLeft())
}
