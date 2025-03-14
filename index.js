const MLP = require('./mlp');
const Value = require('./value');

n = new MLP(3, [4, 4, 1]);
xs = [
  [new Value(2), new Value(3), new Value(-1)],
  [new Value(3), new Value(-1), new Value(5)],
  [new Value(0.5), new Value(1), new Value(1)],
  [new Value(1), new Value(1), new Value(-1)]
];
ys = [
  new Value(1),
  new Value(-1),
  new Value(-1),
  new Value(1)
];

// Forward pass
let ypred = xs.map(x => n.forward(x));
let ilosses = ypred.map((yout, i) => yout.add(ys[i].mul(new Value(-1))).pow(2));
let loss = ilosses.reduce((sum, x) => sum.add(x), new Value(0));
console.log("Starting back propagation...");
while(loss.value > 1e-3) {
  // Backward pass
  const parameters = n.parameters();
  parameters.forEach(p => p.grad = 0);
  loss.backward();

  // Update parameters
  parameters.forEach(p => p.value += (p.grad * -0.1));

  // Forward pass
  ypred = xs.map(x => n.forward(x));
  ilosses = ypred.map((yout, i) => yout.add(ys[i].mul(new Value(-1))).pow(2));
  loss = ilosses.reduce((sum, x) => sum.add(x), new Value(0));
  console.log(loss.value);
}
console.log(ypred.map(x => x.value));