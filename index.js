const MLP = require('./mlp');

n = new MLP(3, [4, 4, 1]);
xs = [
  [2, 3, -1],
  [3, -1, 5],
  [0.5, 1, 1],
  [1, 1, -1]
];
ys = [
  1,
  -1,
  -1,
  1
];

// Forward pass
let ypred = xs.map(x => n.forward(x));
let ilosses = ypred.map((yout, i) => yout.add(ys[i] * -1).pow(2));
let loss = ilosses.reduce((sum, x) => x.add(sum), 0);
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
  ilosses = ypred.map((yout, i) => yout.add(ys[i] * -1).pow(2));
  loss = ilosses.reduce((sum, x) => x.add(sum), 0);
}
console.log("Finished back propagation!");
console.log(ypred.map(x => x.value));