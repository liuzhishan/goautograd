package autograd

import (
	"reflect"
	_ "runtime"
)

type Box struct {
	value interface{}
	node  *VJPNode
	trace int
}

var boxTypes = make(map[reflect.Type]bool)
var boxTypeMappings = make(map[string]reflect.Type)

func registerBox(boxClass reflect.Type, valueType string) {
	boxTypes[boxClass] = true
	boxTypeMappings[valueType] = boxClass
}

func NewBox(value interface{}, trace int, node *VJPNode) *Box {
	if _, ok := boxTypeMappings[getValueType(value)]; ok {
		box := new(Box)
		box.value = value
		box.node = node
		box.trace = trace

		return box
	} else {
		return nil
	}
}

func isBox(x interface{}) bool {
	logInfo("x %v, type: %v", x, reflect.TypeOf(x))
	if _, ok := boxTypes[reflect.TypeOf(x)]; ok {
		return true
	} else {
		return false
	}
}

func trace(startNode *VJPNode, f FuncAny, x float64) (float64, *VJPNode) {
	startBox := NewBox(x, 0, startNode)
	logInfo("x: %v, startBox: %v", x, startBox)
	endBox := f(startBox)
	logInfo("endBox: %v", endBox)
	if isBox(endBox) {
		b := endBox.(*Box)
		logInfo("b: %v", b)
		if b.trace == startBox.trace {
			return b.value.(float64), b.node
		} else {
			return b.value.(float64), nil
		}
	} else {
		return endBox.(float64), nil
	}
}

func primitive(fRaw FuncNumber) FuncAny {
	var fWrapped FuncAny
	fWrapped = func(args ...interface{}) interface{} {
		var argvals []interface{}
		argnums, boxedArgs, trace, nodeConstructorName := findTopBoxedArgs(args...)
		if boxedArgs != nil && len(boxedArgs) > 0 {
			argvals = subvals(args, argnums, getNodeValue(boxedArgs))
			logInfo("argvals: %v", argvals)
			if _, ok := notracePrimitives[nodeConstructorName]; ok {
				return fWrapped(argvals...)
			}

			parents := make([]*VJPNode, 0)
			for _, b := range boxedArgs {
				parents = append(parents, b.node)
			}

			ans := fWrapped(argvals...).(float64)
			node := nodeConstructor(nodeConstructorName, ans, fWrapped, toFloat64(argvals), argnums, parents)

			return NewBox(ans, trace, node)
		} else {
			logInfo("args: %v", args)
			return fRaw(toFloat64(args)...)
		}
	}

	return fWrapped
}

var notracePrimitives = make(map[string]map[string]bool)

func registerNotrace(traceType string, primitiveFun interface{}) {
	if _, ok := notracePrimitives[traceType]; !ok {
		notracePrimitives[traceType] = make(map[string]bool)
	}
	notracePrimitives[traceType][getFuncName(primitiveFun)] = true
}

func notracePrimitive(fRaw FuncAny) FuncAny {
	fWrapped := func(args ...interface{}) interface{} {
		argvals := make([]interface{}, 0)
		for _, v := range args {
			if isBox(v) {
				argvals = append(argvals, v.(Box).value)
			} else {
				argvals = append(argvals, v)
			}
		}

		return fRaw(argvals...)
	}

	return fWrapped
}

func findTopBoxedArgs(args ...interface{}) ([]int, []Box, int, string) {
	argnums := make([]int, 0)
	topBoxes := make([]Box, 0)
	topTrace := -1
	topNodeType := ""

	for i, arg := range args {
		logInfo("i: %d, arg: %v, isBox: %v", i, arg, isBox(arg))
		if isBox(arg) {
			b := arg.(*Box)
			trace := b.trace
			if trace > topTrace {
				argnums = append(argnums, i)
				topBoxes = append(topBoxes, *b)
				topTrace = trace
				topNodeType = getValueType(b)
			} else if trace == topTrace {
				argnums = append(argnums, i)
				topBoxes = append(topBoxes, *b)
			}
		}
	}

	return argnums, topBoxes, topTrace, topNodeType
}

func init() {
	var box Box
	registerBox(reflect.TypeOf(box), getValueType(box))

	var box1 *Box
	registerBox(reflect.TypeOf(box1), getValueType(box1))

	registerBox(reflect.TypeOf(box), "float64")
}
