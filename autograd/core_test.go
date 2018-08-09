package autograd

import (
	"testing"

	"github.com/liuzhishan/gotorch/autograd/num"
)

func TestDefvjp(t *testing.T) {
	gradx := func(args ...float64) FuncNumber {
		return func(args1 ...float64) float64 {
			g := args1[0]
			return g
		}
	}

	grady := func(args ...float64) FuncNumber {
		return func(args1 ...float64) float64 {
			g := args1[0]
			return g
		}
	}

	defvjp(num.Add, gradx, grady)

	logInfo("funcName: %s", getFuncName(num.Add))
	gradAdd := primitiveVjps[getFuncName(num.Add)]
	logInfo("res: %v", gradAdd([]int{}, 1.0, 0.7, 1.8)(1.0))
}
