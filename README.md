# goautograd
goautograd

# introduction
This project is just for learning autograd and Go.

[autograd](https://github.com/HIPS/autograd) is a package that can automatically differentiate native Python and Numpy code. That means, given `f(x) = 1.0 - np.exp(-2.0 * x)`, the expression `grad(f)(2.0)` can automatically compute the gradient of `f` at `x=2.0`.

In order to understand what's going on under the hood, I have implement some core idea in autograd using Go, which includes:

* a map that mapping computation functions to their corresponding gradient functions
* wrap the computation functions so when they are called, they add themselvs to a  list of operations performed
* flag the variables that we are taking the gradient with respect to
* all operations constructed a graph, edges specify the computation order and parameters
* to compute the gradient, we run a preorder traverse on the graph from end node, and compute gradient on each node with the registered gradient functions

Also, Go's support for funtional programming is great. We can have both good side of static typed language and functional programming.

However, autograd supports many operations and data types. I just implement a subset of the functionality that autograd support. There are many things to learn in autograd and Go. I will add some more in the future.