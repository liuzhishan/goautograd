package autograd

// Core functions for autograd.

import (
	"log"
	"reflect"
)

type FuncNumber func(...float64) float64
type FuncGrad func(...float64) FuncNumber
type FuncAny func(...interface{}) interface{}
type VJPMaker func(float64) []float64
type VJPArgnums func([]int, float64, ...float64) VJPMaker

// VJP means vector-Jacobian product functions, or corresponding gradient functions of a function
// VJPNode is a node in the graph. It contains its parent nodes and gradient functions.
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

// nodeConstructor construct a VJPNode during the forward computation. When we encouter
// a registered function that need gradient, we remember all the information needed, which
// includes the function (fKey), parameters (args), result (value), parent nodes.
// The parentArgnums specify which variable we want to get partial gradient to. For example,
// f(x, y) has two input variables, if we just want to compute the partial gradient to x,
// that means df/dx, then parentArgnums would be [0].
// We use fKey to find corresponding gradient functions from the global table that register functions.
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

// Given a function f and a variable x, returns a gradient function and final result.
// Almost all heavy work lies in the function.
// First, we compute forward from the startNode. During computation, we remember all middle
// information that needed for gradient.
// Then, if all things goes fine, we arrive at the endNode, which is the foreard computation
// result. Then we do a backward pass on the graph from the endNode to compute the gradient.
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

// Preorder traverse of the computation graph. At each node, compute gradients, and then
// add the gradients to its parents. When we stop, we have the gradient we want.
// As shown in the picture below. (x+y)*(x-y) is the endNode.
//       x     y
//       |\   /|
//       | \ / |
//       | / \ |
//       |/   \|
//    (x+Y)  (x-y)
//        \  /
//        \ /
//    (x+y)*(x-y)
func backwardPass(g float64, endNode *VJPNode) float64 {
	logInfo("start")
	// A node may have multiple child nodes, which means a node would be used
	// to compute multiple variables. We need to remember it's child counts,
	// so when we run the backward pass, we compute all child nodes before we move
	// to the parent nodes.
	childCounts := make(map[*VJPNode]int)

	// use stack to perform the preorder traverse.
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

// Visit the node, compute gradient value. And add the value to its parent nodes.
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

// Global table that mapping functions to their corresponding gradient functions.
// The number of gradient functions is determined by the variable number.
// For example, f(x, y) = x + y has two partial gradient funtion df/dx and df/dy.
// f(x) = exp(x) has just one gradient function.
var primitiveVjps = make(map[string]VJPArgnums)

func DefvjpArgnums(fKey string, vjpArgnums VJPArgnums) {
	primitiveVjps[fKey] = vjpArgnums
}

// Gegister a function and its corresponding gradient functions.
// In Python, we can use functools.wraps to keep function name unchanged after wrapped.
// But in Go, the wrapped function from one wrapper all has the same name. We can not
// use function name as key after wrapped like in Python. So we need to pass the unique
// fkey to Defvjp. The unique fKey is the function name before wrapped.
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

// Given a function f and variable x, return the gradient result.
func gradOrigin(f FuncAny, x interface{}) interface{} {
	vjp, _ := makeVjp(f, x)
	logInfo("gradOrigin, f: %v, x: %v", getFuncKey(f), x)
	return vjp(1.0)
}

var grad = unaryToNary(gradOrigin)
