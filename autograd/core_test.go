package autograd

import (
	"testing"
)

func TestDefvjp(t *testing.T) {
	logInfo("funcName: %s", getFuncName(Add))
	gradAdd := primitiveVjps[getFuncName(Add)]
	logInfo("res: %v", gradAdd([]int{}, 1.0, 0.7, 1.8)(1.0))
}

func TestGetValueType(t *testing.T) {
	v := 1.0
	logInfo("type: %s", getValueType(v))
}

func TestGrad(t *testing.T) {
	g := grad(Add, 0)(0.7, 1.8)
	logInfo("Add: %s", getFuncKey(Add))
	logInfo("%f", g)
}
