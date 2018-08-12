package autograd

import (
	"log"
	"reflect"
)

type FuncNumber func(...float64) float64
type FuncGrad func(...float64) FuncNumber
type FuncAny func(...interface{}) interface{}
type VJPMaker func(float64) []float64
type VJPArgnums func([]int, float64, ...float64) VJPMaker

type VJPNode struct {
	parents []*VJPNode
	vjp     VJPMaker
	value   float64
}

func NewVJPNode() *VJPNode {
	root := new(VJPNode)
	root.parents = make([]*VJPNode, 0)
	root.vjp = func(g float64) []float64 { return []float64{} }
	root.value = 0.0

	return root
}

func (v VJPNode) init(value float64, fKey string, args []float64, parentArgnums []int, parents []*VJPNode) {
	v.parents = parents
	if vjpmaker, ok := primitiveVjps[fKey]; ok {
		v.vjp = vjpmaker(parentArgnums, value, args...)
	} else {
		log.Fatal("error init node")
	}
}

func nodeConstructor(name string, value float64, fKey string, args []float64, parentArgnums []int, parents []*VJPNode) *VJPNode {
	logInfo("name: %s, f: %s", name, fKey)
	node := NewVJPNode()

	node.value = value
	node.parents = parents
	if vjpmaker, ok := primitiveVjps[fKey]; ok {
		logInfo("find vjpmaker, parentArgnums: %v, value: %v, args: %v", parentArgnums, value, args)
		node.vjp = vjpmaker(parentArgnums, value, args...)
		logInfo("vjp: %v, %v, %f", node.vjp, reflect.TypeOf(node.vjp), node.vjp(1.0))
	} else {
		log.Fatal("error init node")
	}

	return node
}

func makeVjp(f FuncAny, x interface{}) (func(float64) float64, float64) {
	startNode := NewVJPNode()

	logInfo("f: %v, x: %v", getFuncKey(f), x)
	endValue, endNode := trace(startNode, f, x.(float64))
	logInfo("endValue: %v, endNode: %v", endValue, endNode)

	var vjp func(float64) float64
	if endNode == nil {
		vjp = func(g float64) float64 {
			return 0.0
		}
	} else {
		vjp = func(g float64) float64 {
			return backwardPass(g, endNode)
		}
	}

	return vjp, endValue
}

func backwardPass(g float64, endNode *VJPNode) float64 {
	logInfo("start")
	childCounts := make(map[*VJPNode]int)
	stack := []*VJPNode{endNode}

	for {
		if len(stack) == 0 {
			break
		}

		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if _, ok := childCounts[node]; ok {
			childCounts[node] += 1
		} else {
			childCounts[node] = 1
			stack = append(stack, node.parents...)
		}
	}

	logInfo("len(childCounts): %d", len(childCounts))

	outgrad := 0.0
	outgrads := make(map[*VJPNode]float64)
	flags := make(map[*VJPNode]bool)

	outgrads[endNode] = g
	flags[endNode] = true
	childlessNodes := []*VJPNode{endNode}
	for {
		if len(childlessNodes) == 0 {
			break
		}

		node := childlessNodes[len(childlessNodes)-1]
		childlessNodes = childlessNodes[:len(childlessNodes)-1]
		outgrad = outgrads[node]
		logInfo("outgrad: %v", outgrad)
		visit(outgrad, node, outgrads, flags)
		for _, parent := range node.parents {
			// all dependency be computed
			if childCounts[parent] == 1 {
				childlessNodes = append(childlessNodes, parent)
			} else {
				childCounts[parent] -= 1
			}
		}
	}

	logInfo("end")
	return outgrad
}

func visit(outgrad float64, node *VJPNode, outgrads map[*VJPNode]float64, flags map[*VJPNode]bool) {
	logInfo("node: %v, outgrad: %f, node.vjp: %s, node.parents: %v", node, outgrad, getFuncKey(node.vjp), node.parents)
	ingrads := node.vjp(outgrad)
	logInfo("ingrads: %v", ingrads)
	for i, parent := range node.parents {
		if _, ok := outgrads[parent]; !ok {
			outgrads[parent] = 0.0
		}
		outgrads[parent] += ingrads[i]
	}
}

var primitiveVjps = make(map[string]VJPArgnums)

func DefvjpArgnums(fKey string, vjpArgnums VJPArgnums) {
	primitiveVjps[fKey] = vjpArgnums
}

func Defvjp(fKey string, vjpmakers ...FuncGrad) {
	vjpArgnums := func(argnums []int, ans float64, args ...float64) VJPMaker {
		return func(g float64) []float64 {
			arr := make([]float64, 0)
			newArgs := append([]float64{ans}, args...)
			for _, argnum := range argnums {
				arr = append(arr, vjpmakers[argnum](newArgs...)(g))
			}

			return arr
		}
	}

	logInfo("fKey: %s", fKey)
	DefvjpArgnums(fKey, vjpArgnums)
}

func gradOrigin(f FuncAny, x interface{}) interface{} {
	vjp, _ := makeVjp(f, x)
	logInfo("gradOrigin, f: %v, x: %v", getFuncKey(f), x)
	return vjp(1.0)
}

var grad = unaryToNary(gradOrigin)
