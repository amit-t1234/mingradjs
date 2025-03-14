class Value {
  constructor(value, prev=null, op=null, label=null) {
    this._value = value;
    this._prev = new Set(prev);
    this._grad = 0;
    this._backward = function() {};
    this._op = op;
    this._label = label;
  }

  static printTree(t, levels = []) {
    const PFXs = { true: { true: "     ", false: "┃    " }, false: { true: "┗━━━ ", false: "┣━━━ " } };
    const pfx = (p, i) => PFXs[i < levels.length - 1][p];
    console.log(`${levels.map(pfx).join("")}${t.label ? `${t.label}` : ''}|val=${t.value}|grad=${t.grad} ${t.op ? `(operation=${t.op})` : ''}`);
    const prev = new Array(...t.prev);
    prev?.forEach((x, i) => Value.printTree(x, [...levels, i === prev.length-1]));
  }
  
  get value() {
    return this._value;
  }

  get prev() {
    return this._prev;
  }

  get op() {
    return this._op;
  }

  get label() {
    return this._label;
  }

  get grad() {
    return this._grad;
  }

  set grad(grad) {
    this._grad = grad;
  }

  set value(value) {
    this._value = value;
  }
  
  add(value, label) {
    value = value instanceof Value ? value : new Value(value);
    const out = new Value(this.value + value.value, [this, value], '+', label);
    out._backward = () => {
      this.grad += 1 * out.grad;
      value.grad += 1 * out.grad;
    }
    return out;
  }

  mul(value, label) {
    value = value instanceof Value ? value : new Value(value);
    const out = new Value(this.value * value.value, [this, value], '*', label);
    out._backward = () => {
      this.grad += value.value * out.grad;
      value.grad += this.value * out.grad;
    }
    return out;
  }

  tanh(label) {
    const out = new Value(Math.tanh(this.value), [this], 'tanh', label);
    out._backward = () => {
      this.grad += (1 - Math.pow(out.value, 2)) * out.grad;
    }
    return out;
  }

  exp(label) {
    const out = new Value(Math.exp(this.value), [this], 'exp', label);
    out._backward = () => {
      this.grad += Math.exp(this.value) * out.grad;
    }
    return out;
  }

  pow(value, label) {
    if(!(typeof value === 'number')) throw new Error('Exponent must be a number');
    const out = new Value(Math.pow(this.value, value), [this], 'pow', label);
    out._backward = () => {
      this.grad += value * Math.pow(this.value, value - 1) * out.grad;
    }
    return out;
  }

  div(value, label) {
    return this.mul(value.pow(-1), label);
  }

  backward() {
    this.grad = 1;
    const topology = [];
    const visited = new Set();
    const dfs = (node) => {
      if (visited.has(node)) return;
      visited.add(node);
      new Array(...node.prev).forEach(dfs);
      topology.push(node);
    }
    dfs(this);
    topology.reverse();
    topology.forEach(node => node._backward());
  }
}

// Example usage
// const x1 = new Value(2, null, null, 'x1');
// const x2 = new Value(0, null, null, 'x2');
// const w1 = new Value(-3, null, null, 'w1');
// const w2 = new Value(1, null, null, 'w2');
// const b = new Value(6.8813735870195632, null, null, 'b');
// const x1w1 = x1.mul(w1, 'x1w1');
// const x2w2 = x2.mul(w2, 'x2w2');
// const x1w1x2w2 = x1w1.add(x2w2, 'x1w1x2w2');
// const n = x1w1x2w2.add(b, 'n');
// // const y = n.tanh('y');
// const one = new Value(1, null, null, 'one');
// const minusone = one.mul(-1, 'minusone');
// const two = new Value(2, null, null, 'two');
// const ntwo = n.mul(two, 'ntwo');
// const e = ntwo.exp('e');
// const aeone = e.add(one, 'aeone');
// const aeminusone = e.add(minusone, 'minusone');
// y = aeminusone.div(aeone, 'y');
// y.backward();
// printTree(y);

module.exports = Value;