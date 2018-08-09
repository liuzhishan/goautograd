package autograd

func unaryToNary(f interface{}) {
	naryOperator := func(f1 interface{}, argnum []int, naryOpArgs ...interface{}) func(...float64) interface{} {
		naryF := func(args ...float64) interface{} {
			unaryF := func(x interface{}) float64 {
				subargs := subvals(args, argnum, x)
				return f1(subargs...)
			}

			return f(unaryF, x, naryOpArgs...)
		}

		return naryF
	}

	return naryOperator
}
