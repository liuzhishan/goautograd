package autograd

import (
	"log"
)

type FuncNumber func(...float64) float64
type FuncGrad func(...float64) FuncNumber
type VJPMaker func(float64) []float64
type VJPArgnums func([]int, float64, ...float64) VJPMaker

type VJPNode struct {
	parents []*VJPNode
	vjp     VJPMaker
}

func NewVJPNode() *VJPNode {
	root := new(VJPNode)
	root.parents = make([]*VJPNode, 0)
	root.vjp = func(g float64) []float64 { return []float64{} }

	return root
}

func (v VJPNode) init(value float64, f FuncNumber, args []float64, parentArgnums []int, parents []*VJPNode) {
	v.parents = parents
	if vjpmaker, ok := primitiveVjps[getFuncName(f)]; ok {
		v.vjp = vjpmaker(parentArgnums, value, args...)
	} else {
		log.Fatal("error init node")
	}
}

func nodeConstructor(name string, value float64, f FuncNumber, args []float64, parentArgnums []int, parents []*VJPNode) *VJPNode {
	node := NewVJPNode()

	node.parents = parents
	if vjpmaker, ok := primitiveVjps[getFuncName(f)]; ok {
		node.vjp = vjpmaker(parentArgnums, value, args...)
	} else {
		log.Fatal("error init node")
	}

	return node
}

func makeVjp(f FuncNumber, x float64) {
	startNode := NewVJPNode()
	endValue, endNode := trace(startNode, f, x)
	if endNode == nil {
		vjp := func(g float64) float64 {
			return 0.0
		}
	} else {
		vjp := func(g float64) float64 {
			backwardPass(g, endNode)
		}
	}

	return vjp, endValue
}

func backwardPass(g float64, endNode *VJPNode) float64 {
	outgrads := make(map[*VJPNode]float64)
	flags := make(map[*VJPNode]bool)

	childCounts := make(map[*VJPNode]int)
	stack := []*VJPNode{endNode}

	for len(stack) > 0 {
		node, stack := stack[len(stack)-1], stack[:len(stack)-1]
		if v, ok := childCounts[node]; ok {
			childCounts[node] += 1
		} else {
			childCounts[node] = 1
			stack = append(stack, node.parents...)
		}
	}
}

var primitiveVjps = make(map[string]VJPArgnums)

func defvjpArgnums(f FuncNumber, vjpArgnums VJPArgnums) {
	primitiveVjps[getFuncName(f)] = vjpArgnums
}

func defvjp(f FuncNumber, vjpmakers ...FuncGrad) {
	vjpArgnums := func(argnums []int, ans float64, args ...float64) VJPMaker {
		return func(g float64) []float64 {
			arr := make([]float64, 0)
			newArgs := append([]float64{ans}, args...)
			for _, vjp := range vjpmakers {
				arr = append(arr, vjp(newArgs...)(g))
			}

			return arr
		}
	}

	defvjpArgnums(f, vjpArgnums)
}
