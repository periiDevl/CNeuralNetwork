#include "MNISTLoader.h"

static int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) | ((int)c2 << 16) | ((int)c3 << 8) | c4;
}

double** loadMNISTImages(const char* path, int* outCount) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("ERROR: Cannot open %s\n", path); return NULL; }

    int magic, count, rows, cols;
    fread(&magic, sizeof(int), 1, f);  magic = reverseInt(magic);
    fread(&count, sizeof(int), 1, f);  count = reverseInt(count);
    fread(&rows,  sizeof(int), 1, f);  rows  = reverseInt(rows);
    fread(&cols,  sizeof(int), 1, f);  cols  = reverseInt(cols);

    int pixels = rows * cols;
    double** images = (double**)malloc(count * sizeof(double*));

    for (int i = 0; i < count; i++) {
        images[i] = (double*)malloc(pixels * sizeof(double));
        for (int p = 0; p < pixels; p++) {
            unsigned char pixel = 0;
            fread(&pixel, sizeof(unsigned char), 1, f);
            images[i][p] = pixel / 255.0;
        }
    }

    fclose(f);
    *outCount = count;
    return images;
}

int* loadMNISTLabels(const char* path, int* outCount) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("ERROR: Cannot open %s\n", path); return NULL; }

    int magic, count;
    fread(&magic, sizeof(int), 1, f);  magic = reverseInt(magic);
    fread(&count, sizeof(int), 1, f);  count = reverseInt(count);

    int* labels = (int*)malloc(count * sizeof(int));
    for (int i = 0; i < count; i++) {
        unsigned char label = 0;
        fread(&label, sizeof(unsigned char), 1, f);
        labels[i] = (int)label;
    }

    fclose(f);
    *outCount = count;
    return labels;
}

double** toOneHot(int* labels, int count) {
    double** oneHot = (double**)malloc(count * sizeof(double*));
    for (int i = 0; i < count; i++) {
        oneHot[i] = (double*)calloc(10, sizeof(double));
        oneHot[i][labels[i]] = 1.0;
    }
    return oneHot;
}

void freeMNISTImages(double** images, int count) {
    for (int i = 0; i < count; i++) free(images[i]);
    free(images);
}
void freeMNISTLabels(int* labels) { free(labels); }
void freeOneHot(double** oneHot, int count) {
    for (int i = 0; i < count; i++) free(oneHot[i]);
    free(oneHot);
}
