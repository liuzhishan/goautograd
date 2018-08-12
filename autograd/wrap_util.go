package autograd

func unaryToNary(unaryOperator func(FuncAny, interface{}) interface{}) func(FuncAny, interface{}) FuncAny {
	naryOperator := func(f FuncAny, argnum interface{}) FuncAny {
		naryF := func(args ...interface{}) interface{} {
			unaryF := func(x ...interface{}) interface{} {
				logInfo("args: %v, argnum: %d, x: %v", args, argnum, x)
				subargs := subvals(args, []int{argnum.(int)}, x)
				logInfo("unaryF, x: %v, args: %v, subargs: %v", x, args, subargs)
				return f(subargs...)
			}

			logInfo("naryF, argnum: %v, args: %v", argnum, args)
			return unaryOperator(unaryF, args[argnum.(int)])
		}

		return naryF
	}

	return naryOperator
}
