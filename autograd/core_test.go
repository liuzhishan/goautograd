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

func TestGradAdd(t *testing.T) {
	g := grad(Add, 0)(0.7, 1.8)
	logInfo("Add: %s", getFuncKey(Add))
	logInfo("%f", g)
}

func TestGradSub(t *testing.T) {
	g := grad(Sub, 1)(0.7, 1.8)
	logInfo("Sub: %s", getFuncKey(Sub))
	logInfo("%f", g)
}

func TestGradMul(t *testing.T) {
	gx := grad(Mul, 0)(0.7, 2.8)
	logInfo("Mul: %s", getFuncKey(Mul))
	logInfo("gx: %f", gx)
	logInfo("===================")

	gy := grad(Mul, 1)(0.7, 2.8)
	logInfo("Mul: %s", getFuncKey(Mul))
	logInfo("gy: %f", gy)
}

func TestGradMulAddx(t *testing.T) {
	f := func(args ...interface{}) interface{} {
		x, y := args[0], args[1]
		a := Add([]interface{}{x, y}...)
		b := Sub([]interface{}{x, y}...)
		c := Mul([]interface{}{a, b}...)
		d := Add([]interface{}{x, c}...)

		return d
	}

	// gx = 2.4
	gx := grad(f, 0)(0.7, 1.8)
	logInfo("Mul: %s", getFuncKey(f))
	logInfo("gx: %f", gx)
}

func TestGradMulAddy(t *testing.T) {
	f := func(args ...interface{}) interface{} {
		x, y := args[0], args[1]
		a := Add([]interface{}{x, y}...)
		b := Sub([]interface{}{x, y}...)
		c := Mul([]interface{}{a, b}...)
		d := Add([]interface{}{y, c}...)

		return d
	}

	// gx = -2.6
	gy := grad(f, 1)(0.7, 1.8)
	logInfo("Mul: %s", getFuncKey(f))
	logInfo("gy: %f", gy)
}
