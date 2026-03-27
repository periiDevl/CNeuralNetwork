#include "NetworkIO.h"
#include <stdio.h>
#include <stdlib.h>

void saveNetwork(NeuralNetwork* network, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Could not open file for saving.\n");
        return;
    }

    // 1. Write number of layers
    fwrite(&network->numLayers, sizeof(unsigned long long int), 1, file);

    // 2. Write the size of each layer
    for (size_t i = 0; i < network->numLayers; i++) {
        fwrite(&network->layers[i].numNeurons, sizeof(unsigned long long int), 1, file);
    }

    // 3. Write Biases and Weights for every neuron
    for (size_t i = 0; i < network->numLayers; i++) {
        for (size_t j = 0; j < network->layers[i].numNeurons; j++) {
            Neuron* n = &network->layers[i].neurons[j];
            
            // Write bias
            fwrite(&n->bias, sizeof(double), 1, file);
            
            // Write weight size and weights array
            fwrite(&n->weightsSize, sizeof(unsigned long long int), 1, file);
            if (n->weightsSize > 0) {
                fwrite(n->weights, sizeof(double), n->weightsSize, file);
            }
        }
    }
    
    fclose(file);
    printf("Network successfully saved to %s\n", filename);
}

void loadNetwork(NeuralNetwork* network, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file for loading.\n");
        return;
    }

    // 1. Read number of layers and create network
    unsigned long long int numLayers;
    fread(&numLayers, sizeof(unsigned long long int), 1, file);
    createNeuralNetwork(network, numLayers);

    // 2. Read layer sizes and allocate neurons
    unsigned long long int* layerSizes = malloc(numLayers * sizeof(unsigned long long int));
    for (size_t i = 0; i < numLayers; i++) {
        fread(&layerSizes[i], sizeof(unsigned long long int), 1, file);
        setNeuronsSize(&network->layers[i], layerSizes[i]);
    }

    // Link and set parameters (allocates weight arrays)
    linkLayers(network);
    setLayersParam(network);

    // 3. Read Biases and Weights
    for (size_t i = 0; i < numLayers; i++) {
        for (size_t j = 0; j < layerSizes[i]; j++) {
            Neuron* n = &network->layers[i].neurons[j];
            
            fread(&n->bias, sizeof(double), 1, file);
            
            unsigned long long int wSize;
            fread(&wSize, sizeof(unsigned long long int), 1, file);
            
            if (wSize > 0) {
                fread(n->weights, sizeof(double), wSize, file);
            }
        }
    }

    free(layerSizes);
    fclose(file);
    printf("Network successfully loaded from %s\n", filename);
}