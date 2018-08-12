package autograd

import ()

func AddOrigin(args ...float64) float64 {
	logInfo("args: %v", args)
	a, b := args[0], args[1]
	return a + b
}

var Add = primitive(AddOrigin)

var gradx = func(args ...float64) FuncNumber {
	return func(args1 ...float64) float64 {
		g := args1[0]
		return g
	}
}

var grady = func(args ...float64) FuncNumber {
	return func(args1 ...float64) float64 {
		g := args1[0]
		return g
	}
}

func init() {
	Defvjp(Add, gradx, grady)
}
