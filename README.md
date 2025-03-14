# Neural Network Project

This project implements a simple neural network using JavaScript. It includes a multi-layer perceptron (MLP) and a custom value class for automatic differentiation.

## Installation

To install and set up the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/amit-t1234/mingradjs.git
cd mingradjs

# Install dependencies (if any)
npm install
```

## Usage

To use the project, you can run the `index.js` file which demonstrates training a neural network:

```bash
# Run the example
node index.js
```

## Code Overview

### `mlp.js`

This file contains the implementation of the multi-layer perceptron (MLP) which includes the `Neuron`, `Layer`, and `MLP` classes.

- **Neuron**: Represents a single neuron in the network. It includes methods for forward and backward passes.
- **Layer**: Represents a layer of neurons. It includes methods to perform forward and backward passes for the entire layer.
- **MLP**: Represents the multi-layer perceptron. It includes methods to perform forward and backward passes for the entire network and to update the parameters.

### `value.js`

This file contains the `Value` class which supports operations like addition, multiplication, and automatic differentiation.

- **Value**: A custom class that represents a value in the computation graph. It supports basic arithmetic operations and tracks gradients for automatic differentiation.

### `index.js`

This file demonstrates how to create an MLP, perform forward and backward passes, and update the parameters to train the network.

- **Example Usage**: Shows how to initialize the MLP, perform training iterations, and print the results.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- This project is JS implementation of the https://github.com/karpathy/micrograd library.
- Special thanks to Andrej Karpathy!

## Contact

For any questions or feedback, please open an issue on the repository or contact the project maintainer at [amitthakurashwani@gmail.com].