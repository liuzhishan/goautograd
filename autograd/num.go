package autograd

import ()

// Some simple numpy function and their gradient functions.
func AddOrigin(args ...float64) float64 {
	logInfo("AddOrigin, args: %v", args)
	a, b := args[0], args[1]
	return a + b
}

var Add = primitive(AddOrigin)

var gradxAdd = func(args ...float64) FuncNumber {
	return func(args1 ...float64) float64 {
		g := args1[0]
		return g
	}
}

var gradyAdd = func(args ...float64) FuncNumber {
	return func(args1 ...float64) float64 {
		g := args1[0]
		logInfo("gradxAdd, args: %v, args1: %v, res: %f", args, args1, 1.0)
		return g
	}
}

func SubOrigin(args ...float64) float64 {
	logInfo("SubOrigin, args: %v", args)
	a, b := args[0], args[1]
	return a - b
}

var Sub = primitive(SubOrigin)

var gradxSub = func(args ...float64) FuncNumber {
	return func(args1 ...float64) float64 {
		g := args1[0]
		logInfo("gradxSub, args: %v, args1: %v, res: %f", args, args1, 1.0)
		return g
	}
}

var gradySub = func(args ...float64) FuncNumber {
	return func(args1 ...float64) float64 {
		g := args1[0]
		return -g
	}
}

func MulOrigin(args ...float64) float64 {
	logInfo("MulOrigin, args: %v", args)
	a, b := args[0], args[1]
	return a * b
}

var Mul = primitive(MulOrigin)

var gradxMul = func(args ...float64) FuncNumber {
	return func(args1 ...float64) float64 {
		g := args1[0]
		logInfo("gradxMul, args: %v, args1: %v, res: %f", args, args1, g*args[2])
		return g * args[2]
	}
}

var gradyMul = func(args ...float64) FuncNumber {
	return func(args1 ...float64) float64 {
		g := args1[0]
		logInfo("gradyMul, args: %v, args1: %v, res: %f", args, args1, g*args[1])
		return g * args[1]
	}
}

func init() {
	Defvjp(getFuncKey(AddOrigin), gradxAdd, gradyAdd)
	Defvjp(getFuncKey(SubOrigin), gradxSub, gradySub)
	Defvjp(getFuncKey(MulOrigin), gradxMul, gradyMul)
}
