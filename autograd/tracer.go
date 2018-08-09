package autograd

type Box struct {
	value interface{}
	node  VJPNode
	trace int
}

var boxTypes = make(map[string]bool)
var boxTypeMappings = make(map[string]interface{})

func registerBox(boxClass interface{}, boxClassName string, valueType string) {
	boxTypes[boxClassName] = true
	boxTypeMappings[valuetype] = boxClass
	boxTypeMappings[boxClassName] = boxClass
}

func newBox(value interface{}, trace int, node VJPNode) *Box {
	box := new(boxTypeMappings[getValueType(value)])
	box.value = value
	box.node = node
	box.trace = trace

	return box
}

func trace(startNode VJPNode, f FuncNumber, x float64) (float64, VJPNode) {
	startBox := newBox(x, 0, startNode)
	endBox := f(startBox)
	if isBox(endBox) && endBox.trace == startBox.trace {
		return endBox.value, endBox.node
	} else {
		return endBox, nil
	}
}
