package autograd

import ()

type FuncNumber func(...float64) float64
type FuncGrad func(...float64) FuncNumber
type VJPMaker func(float64) []float64
type VJPArgnums func([]int, float64, ...float64) VJPMaker

type VJPNode struct {
	parents []*VJPNode
	vjp     []FuncGrad
}

func NewVJPNode() *VJPNode {
	root := new(VJPNode)
	root.parents = make([]*VJPNode, 0)
	root.vjp = make([]FuncGrad, 0)

	return root
}

func (v VJPNode) init(value float64, f FuncNumber) {

}

func makeVjp(f FuncNumber, x float64) {
	startNode := newVJPNode()
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
