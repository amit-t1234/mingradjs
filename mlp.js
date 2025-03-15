const Value = require('./value');

class Neuron {
  constructor(nin) {
    this.weights = Array.from({ length: nin }, () => new Value(Math.random() * 2 - 1));
    this.bias = new Value(Math.random() * 2 - 1);
  }

  forward(inputs) {
    inputs = inputs.map(input => input instanceof Value ? input : new Value(input));
    const sum = inputs.reduce((sum, input, i) => sum.add(input.mul(this.weights[i])), this.bias);
    return sum.tanh();
  }

  parameters() {
    return this.weights.concat(this.bias);
  }
}

class Layer {
  constructor(nin, nout) {
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin));
  }

  forward(inputs) {
    const outs = this.neurons.map(neuron => neuron.forward(inputs));
    return outs.length === 1 ? outs[0] : outs;
  }

  parameters() {
    return this.neurons.reduce((params, neuron) => params.concat(neuron.parameters()), []);
  }
}

class MLP {
  constructor(nin, nouts) {
    this.layers = nouts.reduce((layers, nout) => {
      const layer = new Layer(nin, nout);
      layers.push(layer);
      nin = nout;
      return layers;
    }, []);
  }

  forward(inputs) {
    return this.layers.reduce((inputs, layer) => layer.forward(inputs), inputs);
  }

  parameters() {
    return this.layers.reduce((params, layer) => params.concat(layer.parameters()), []);
  }
}

module.exports = MLP;