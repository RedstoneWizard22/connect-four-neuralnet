import { multiply } from 'mathjs';

type ActivationFunction = (x: number) => number;

type InputLayer = {
  activation: ActivationFunction;
  size: number;
  values: number[];
  weights: number[][];
  connectsTo: HiddenLayer;
};

type HiddenLayer = {
  activation: ActivationFunction;
  size: number;
  values: number[];
  biases: number[];
  weights: number[][];
  connectsTo: HiddenLayer | OutputLayer;
};

type OutputLayer = {
  activation: ActivationFunction;
  size: number;
  values: number[];
  biases: number[];
};

class MultilayerPerceptron {
  _inputSize: number;
  _outputSize: number;
  _hiddenLayerCount: number;
  _hiddenLayerSize: number;

  inputLayer: InputLayer;
  outputLayer: OutputLayer;
  hiddenLayers: HiddenLayer[];

  constructor(
    inputSize: number,
    outputSize: number,
    hiddenLayerCount: number,
    hiddenLayerSize: number,
    activationFunction: ActivationFunction,
    inputActivationFunction?: ActivationFunction,
    outputActivationFunction?: ActivationFunction
  ) {
    // Check if the number of hidden layers is valid
    if (hiddenLayerCount < 1) {
      throw new Error('At least one hidden layer is required');
    }

    // Check if the layer sizes are valid
    if (inputSize < 1 || outputSize < 1 || hiddenLayerSize < 1) {
      throw new Error('Layer sizes must be greater than zero');
    }

    /// Create all the layers
    this.outputLayer = {
      activation: outputActivationFunction || activationFunction,
      size: outputSize,
      values: Array(outputSize).fill(0) as number[],
      biases: Array(outputSize).fill(0) as number[],
    };

    this.hiddenLayers = [];
    for (let i = 0; i < hiddenLayerCount; i++) {
      const connectsTo = this.hiddenLayers[0] || this.outputLayer;

      this.hiddenLayers.unshift({
        activation: activationFunction,
        size: hiddenLayerSize,
        values: Array(hiddenLayerSize).fill(0) as number[],
        biases: Array(hiddenLayerSize).fill(0) as number[],
        weights: Array(connectsTo.size).fill(
          Array(hiddenLayerSize).fill(0)
        ) as number[][],
        connectsTo,
      });
    }

    this.inputLayer = {
      activation: inputActivationFunction || activationFunction,
      size: inputSize,
      values: Array(inputSize).fill(0) as number[],
      weights: Array(hiddenLayerSize).fill(
        Array(inputSize).fill(0)
      ) as number[][],
      connectsTo: this.hiddenLayers[0],
    };

    // Set parameters
    this._inputSize = inputSize;
    this._outputSize = outputSize;
    this._hiddenLayerCount = hiddenLayerCount;
    this._hiddenLayerSize = hiddenLayerSize;
  }

  /** Computes values for neurons in all layers */
  computeNeurons(): void {
    // Compute hidden layer 1
    this.hiddenLayers[0].values = multiply(
      this.inputLayer.weights,
      this.inputLayer.values
    )[0].map((value, index) => {
      return this.hiddenLayers[0].activation(
        value + this.hiddenLayers[0].biases[index]
      );
    });

    // Compute hidden layers 2+
    for (let i = 1; i < this._hiddenLayerCount; i++) {
      this.hiddenLayers[i].values = multiply(
        this.hiddenLayers[i - 1].weights,
        this.hiddenLayers[i - 1].values
      )[0].map((value, index) => {
        return this.hiddenLayers[i].activation(
          value + this.hiddenLayers[i].biases[index]
        );
      });
    }

    // Compute output layer
    this.outputLayer.values = multiply(
      this.hiddenLayers[this._hiddenLayerCount - 1].weights,
      this.hiddenLayers[this._hiddenLayerCount - 1].values
    )[0].map((value, index) => {
      return this.outputLayer.activation(
        value + this.outputLayer.biases[index]
      );
    });
  }
}

export default MultilayerPerceptron;
