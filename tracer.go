package autograd

import (
	"reflect"
)

type Box struct {
	value interface{}
	node  VJPNode
	trace int
}

var boxTypes = make(map[string]bool)
var boxTypeMappings = make(map[string]reflect.Type)

var typeRegistry = make(map[string]reflect.Type)

func registerBox(boxClass interface{}, valueType string) {
	boxTypes[boxClassName] = true
	boxTypeMappings[valueType] = reflect.TypeOf(boxClass)
	boxTypeMappings[getValueType(boxClass)] = reflect.TypeOf(boxClass)
}

func NewBox(value interface{}, trace int, node VJPNode) *Box {
	box := new(boxTypeMappings[getValueType(value)])
	box.value = value
	box.node = node
	box.trace = trace

	return box
}

func trace(startNode *VJPNode, f FuncNumber, x float64) (float64, *VJPNode) {
	startBox := newBox(x, 0, startNode)
	endBox := f(startBox)
	if isBox(endBox) && endBox.trace == startBox.trace {
		return endBox.value, endBox.node
	} else {
		return endBox, nil
	}
}

func primitive(fRaw interface{}) interface{} {
	fWrapped := func(args ...interface{}) interface{} {
		argnums, boxedArgs, trace, nodeConstructorName := findTopBoxedArgs(args)
		if boxedArgs != nil {
			argvals := subvals(args, argnums, boxedArgs)
			if v, ok := notracePrimitives[nodeConstructorName]; ok {
				return fWrapped()
			}

			parents := make([]*VJPNode, 0)
			for _, b := range boxedArgs {
				parents = append(parents, b)
			}

			ans := fWrapped(argvals...)
			node := nodeConstructor(nodeConstructorName, ans, fWrapped, argvals, argnums, parents)

			return NewBox(ans, trace, node)
		} else {
			return fWrapped(argvals...)
		}
	}

	return fWrapped
}

var notracePrimitives = make(map[string]map[string]bool)

func registerNotrace(traceType string, primitiveFun interface{}) {
	notracePrimitives[traceType][getFuncName(primitiveFun)] = true
}

func notracePrimitive(fRaw interface{}) {
	fWrapped := func(args ...interface{}) interface{} {
		argvals := make([]float64, 0)
		for _, v := range args {
			if ixBox(v) {
				argvals = append(argvals, v.value)
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
		if ixBox(arg) {
			trace := arg.trace
			if trace > topTrace {
				argnums = append(argnums, i)
				topBoxes = append(topBoxes, arg)
				topTrace = trace
				topNodeType = getValueType(arg)
			} else if trace == topTrace {
				argnums = append(argnums, i)
				topBoxes = append(topBoxes, arg)
			}
		}
	}

	return argnums, topBoxes, topTrace, topNodeType
}
