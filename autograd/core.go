package autograd

import ()

type FuncGrad func(...float64) float64

type VJPNode struct {
	parents []*VJPNode
	vjp     FuncGrad
}

func NewVJPNode() VJPNode {
	root := VJPNode(make([]*VJPNode), func(x float64) float64 { return x })
	return root
}

func (v VJPNode) init(value float64, f FuncGrad) {

}

var primitiveVjps = make(map[string][]FuncGrad)

func defjvpArgnums(f FuncGrad, vjpMaker []FuncGrad) {

}
