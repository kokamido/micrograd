
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, label, _children=(), _op='', show_grads=False):
        self.data = float(data)
        self.label = label
        self.grad = 0
        self.show_grads = show_grads
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, label='?')
        out = Value(self.data + other.data,  f'{self.label} + {other.label}', (self, other), '+', show_grads=True)

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __sub__(self, other):
        print('sub')
        other = other if isinstance(other, Value) else Value(other, label='?')
        out = Value(self.data - other.data,  f'{self.label} - {other.label}', (self, other), '-', show_grads=True)

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label='?')
        out = Value(self.data * other.data,  f'{self.label} * {other.label}', (self, other), '*', show_grads=True)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other,  f'({self.label}) ^ {other}', (self,), f'^{other}', show_grads=True)

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
