#ifndef H_BACKPROP
#define H_BACKPROP
#include"NeuralNetwork.h"

double ELUderivative(double x);
double sigmoidDerivative(double x);
void computeAllNetworkGradients(NeuralNetwork* network, double* expectedOutputs);
double calculateNetworkCost(NeuralNetwork* network, double* expectedOutputs);
double learnFromSample(NeuralNetwork* network, double* inputs, double* expectedOutputs, double learningRate);
void clearGradients(NeuralNetwork* network);
#endif