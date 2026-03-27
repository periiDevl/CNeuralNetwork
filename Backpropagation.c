#include"Backpropagation.h"

double ELUderivative(double x) {
    if (x > 0) {
        return 1.0;
    } else {
        return 1.0 * exp(x);
    }
}

double sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double getIndvCost(double expected, double output){
    return (expected-output)*(expected-output);
}

double getIndvCostDeriv(double expected, double output){
    return 2*(output-expected);
}

double calculateNetworkCost(NeuralNetwork* network, double* expectedOutputs) {
    double totalCost = 0.0;
    size_t outputLayer = network->numLayers - 1;
    
    for (size_t neuron = 0; neuron < network->layers[outputLayer].numNeurons; neuron++) {
        double output = network->layers[outputLayer].neurons[neuron].val;
        double expected = expectedOutputs[neuron];
        totalCost += getIndvCost(expected, output);
    }
    
    return totalCost / network->layers[outputLayer].numNeurons;
}

void layerBackward(Layer* currentLayer, Layer* nextLayer, double* currentDeltas, double* nextDeltas) {
    if (currentLayer == NULL || nextLayer == NULL) return;
    for (size_t i = 0; i < currentLayer->numNeurons; i++) {
        for (size_t j = 0; j < nextLayer->numNeurons; j++) {
            double input_activation = currentLayer->neurons[i].val;
            double gradient = input_activation * nextDeltas[j];
            currentLayer->neurons[i].weightsGradients[j] = gradient;
        }
    }
    for (size_t j = 0; j < nextLayer->numNeurons; j++) {
        nextLayer->neurons[j].biasGradient = nextDeltas[j];
    }
    for (size_t i = 0; i < currentLayer->numNeurons; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < nextLayer->numNeurons; j++) {
            double weight = currentLayer->neurons[i].weights[j];
            sum += nextDeltas[j] * weight;
        }
        double z = currentLayer->neurons[i].Z;
        currentDeltas[i] = sum * sigmoidDerivative(z);
    }
}

void computeAllNetworkGradients(NeuralNetwork* network, double* expectedOutputs) {
    double** deltas = (double**)malloc(network->numLayers * sizeof(double*));
    for (size_t layer = 0; layer < network->numLayers; layer++) {
        deltas[layer] = (double*)calloc(network->layers[layer].numNeurons, sizeof(double));
    }
    size_t outputLayer = network->numLayers - 1;
    for (size_t neuron = 0; neuron < network->layers[outputLayer].numNeurons; neuron++) {
        double z = network->layers[outputLayer].neurons[neuron].Z;
        double output = network->layers[outputLayer].neurons[neuron].val;
        double expected = expectedOutputs[neuron];
        
        double costDeriv = getIndvCostDeriv(expected, output);
        double activationDeriv = sigmoidDerivative(z);
        deltas[outputLayer][neuron] = costDeriv * activationDeriv;
    }
    for (size_t neuron = 0; neuron < network->layers[outputLayer].numNeurons; neuron++) {
        network->layers[outputLayer].neurons[neuron].biasGradient = deltas[outputLayer][neuron];
    }
    for (int layer = network->numLayers - 2; layer >= 0; layer--) {
        layerBackward(&network->layers[layer], &network->layers[layer + 1], 
                     deltas[layer], deltas[layer + 1]);
    }
    for (size_t layer = 0; layer < network->numLayers; layer++) {
        free(deltas[layer]);
    }
    free(deltas);
}

void updateNetworkWeights(NeuralNetwork* network, double learningRate) {
    for (size_t layer = 0; layer < network->numLayers - 1; layer++) {
        for (size_t neuron = 0; neuron < network->layers[layer].numNeurons; neuron++) {
            for (size_t nextNeuron = 0; nextNeuron < network->layers[layer + 1].numNeurons; nextNeuron++) {
                double gradient = network->layers[layer].neurons[neuron].weightsGradients[nextNeuron];
                network->layers[layer].neurons[neuron].weights[nextNeuron] -= learningRate * gradient;
            }
        }
    }
    for (size_t layer = 1; layer < network->numLayers; layer++) {
        for (size_t neuron = 0; neuron < network->layers[layer].numNeurons; neuron++) {
            double gradient = network->layers[layer].neurons[neuron].biasGradient;
            network->layers[layer].neurons[neuron].bias -= learningRate * gradient;
        }
    }
}

double learnFromSample(NeuralNetwork* network, double* inputs, double* expectedOutputs, double learningRate) {
    for (size_t i = 0; i < network->layers[0].numNeurons; i++) {
        network->layers[0].neurons[i].val = inputs[i];
    }
    forwardPass(network);
    double cost = calculateNetworkCost(network, expectedOutputs);
    computeAllNetworkGradients(network, expectedOutputs);
    updateNetworkWeights(network, learningRate);
    return cost;
}