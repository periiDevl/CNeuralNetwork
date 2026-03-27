#ifndef NETWORK_IO_H
#define NETWORK_IO_H
#include "NeuralNetwork.h"

// Saves the entire network to a binary file
void saveNetwork(NeuralNetwork* network, const char* filename);

// Allocates and loads a network from a binary file
void loadNetwork(NeuralNetwork* network, const char* filename);

#endif