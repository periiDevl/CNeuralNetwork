#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H
#include <stdio.h>
#include <stdlib.h>

double** loadMNISTImages(const char* path, int* outCount);
int*     loadMNISTLabels(const char* path, int* outCount);
double** toOneHot(int* labels, int count);
void     freeMNISTImages(double** images, int count);
void     freeMNISTLabels(int* labels);
void     freeOneHot(double** oneHot, int count);

#endif
