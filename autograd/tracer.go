package autograd

type JVPNode struct {
	g func(interface{}) interface{}
}

// func NewJVPNode() JVPNode {
// 	node := JVPNode()
// 	return node
// }

var primitiveJvps = make(map[string]interface{})
