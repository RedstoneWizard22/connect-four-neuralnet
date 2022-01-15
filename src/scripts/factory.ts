import type MultilayerPerceptron from './neuralnet';

/** Randomizes weights and biases for a MultilayerPerceptron */
function randomizeMLP(mlp: MultilayerPerceptron): void {
  // Randomize input layer weights
  for (let i = 0; i < mlp.inputLayer.size; i++) {
    for (let j = 0; j < mlp.hiddenLayers[0].size; j++) {
      mlp.inputLayer.weights[j][i] = Math.random() * 2 - 1;
    }
  }

  // Randomize hidden layer weights
  for (let i = 0; i < mlp.hiddenLayers.length; i++) {
    if (i == mlp._hiddenLayerCount - 1) {
      for (let j = 0; j < mlp.hiddenLayers[i].size; j++) {
        for (let k = 0; k < mlp.outputLayer.size; k++) {
          mlp.hiddenLayers[i].weights[k][j] = Math.random() * 2 - 1;
        }
      }
      continue;
    }
    for (let j = 0; j < mlp.hiddenLayers[i].size; j++) {
      for (let k = 0; k < mlp.hiddenLayers[i + 1].size; k++) {
        mlp.hiddenLayers[i].weights[k][j] = Math.random() * 2 - 1;
      }
    }
  }

  // Randomize hidden layer biases
  for (let i = 0; i < mlp.hiddenLayers.length; i++) {
    for (let j = 0; j < mlp.hiddenLayers[i].size; j++) {
      mlp.hiddenLayers[i].biases[j] = Math.random() * 2 - 1;
    }
  }

  // Randomize output layer biases
  for (let i = 0; i < mlp.outputLayer.size; i++) {
    mlp.outputLayer.biases[i] = Math.random() * 2 - 1;
  }
}

export { randomizeMLP };
