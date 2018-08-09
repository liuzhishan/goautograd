package autograd

import (
	"testing"
)

func TestMapFunc(t *testing.T) {
	m := make(map[string]FuncGrad)
	m["add"] = func(arr ...float64) float64 { return sum(arr...) }
	logInfo("len(m): %d", len(m))
	logInfo("res: %f", m["add"](3.0, 4.0))
}
