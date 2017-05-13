#include "cnn.h"
#include <stdio.h>
#include <stdlib.h>

float relu(float x)
{
    if (x <= 0)
        return 0;
    else 
        return x;
}


float locateSub(float **a, int m, int n, int stride, int kernelSize)
{
    int i, j;
    int max = a[m * stride][n * stride];
    for (i = m * stride; i < m * stride + kernelSize; ++i) {
        for (j = n * stride; j < n * stride + kernelSize; ++j) {
            if (a[i][j] > max)
                max = a[i][j];
        }
    }
    return max;
}

void maxPooling(poolLayer *pool)
{
    int inputSize = pool->w1;
    int s = pool->stride;
    int kernelSize = pool->kernelSize;
    int pooledSize = (inputSize - kernelSize) / s + 1;
    int i, j, l;
    for (l = 0; l < pool->inChannels; ++l) {
        for (i = 0; i < pooledSize; ++i) {
            for (j = 0; j < pooledSize; ++j) {
                pool->pooledData[l][i][j] = locateSub(pool->inputData[l], i, j, pool->stride, pool->kernelSize);
            }
        }
    }
}

float averageHelper(float **a, int m, int n, int stride, int kernelSize)
{
    int i, j;
    float sum = 0;
    for (i = m * stride; i < m * stride + kernelSize; ++i) {
        for (j = n * stride; j < n * stride + kernelSize; ++j) {
            sum += a[i][j];
        }
    }
    return sum / kernelSize / kernelSize;
}

void averagePooling(poolLayer *pool)
{
    int inputSize = pool->w1;
    int s = pool->stride;
    int kernelSize = pool->kernelSize;
    int pooledSize = (inputSize - kernelSize) / s + 1;
    int i, j, l;
    for (l = 0; l < pool->inChannels; ++l) {
        for (i = 0; i < pooledSize; ++i) {
            for (j = 0; j < pooledSize; ++j) {
                pool->pooledData[l][i][j] = averageHelper(pool->inputData[l], i, j, pool->stride, pool->kernelSize);
            }
        }
    }
}

void conv3D(convLayer *conv)
{
    int l, m, n;
    float sum;
    int outSize = (conv->w1 - conv->kernelSize + 2 * conv->padding) / conv->stride + 1;
    for (l = 0; l < conv->numOutput; ++l) {
        for (m = 0; m < outSize; ++m) {
            for (n = 0; n < outSize; ++n) {
                sum = conv_helper(conv, conv->inChannels, m, n, conv->stride, l);
                conv->featureMap[l][m][n] = relu(sum + conv->biasData[l]);
            }
        }
    }
}

float conv_helper(convLayer *conv, int channels, int loc_m, int loc_n, int stride, int nth_filter)
{
    int c, m, n;
    float sum = 0;
    for (c = 0; c < channels; ++c) {
        for (m = loc_m * stride; m < loc_m * stride + conv->kernelSize; ++m) {
            for (n = loc_n * stride; n < loc_n * stride + conv->kernelSize; ++n) {
                sum += conv->inputData[c][m][n] * conv->filters[nth_filter][c][m-loc_m*stride][n-loc_n*stride];
            }
        }
    }
    return sum;
}

int cl_init(convLayer *conv, int inputSize, int kernelSize, int inChannels, int numOutput, int padding, int stride)
{
    conv->h1 = conv->w1 = inputSize;
    conv->kernelSize = kernelSize;
    conv->inChannels = inChannels;
    conv->numOutput = numOutput;
    conv->padding = padding;
    conv->stride = stride;
    return (inputSize - kernelSize + 2 * padding ) / stride + 1;
}

int pl_init(poolLayer *pl, float ***inputData, int inputSize, int kernelSize, int inChannels, int stride)
{
    pl->inputData = inputData;
    pl->w1 = pl->h1 = inputSize;
    pl->kernelSize = kernelSize;
    pl->inChannels = inChannels;
    pl->stride = stride;
    return (inputSize - kernelSize) / stride + 1;
}
void pooled2fc(poolLayer pool, float ****pooled2fcWeights, fcLayer *fc)
{
    int l, k, i, j;
    int pooledSize = (pool.w1 - pool.kernelSize) / pool.stride + 1;
    for (l = 0; l < fc->outputNum; ++l) {
        float sum = 0;
        for (k = 0; k < pool.inChannels; ++k) {
            for (i = 0; i < pooledSize; ++i) {
                for (j = 0; j < pooledSize; ++j) {
                    sum += (pool.pooledData)[k][i][j] * pooled2fcWeights[l][k][i][j];
                }
            }
        }
        fc->outputData[l] = relu(sum + fc->biasData[l]);
    }
}

void fc2fc(fcLayer *fc)
{
    int i, j;
    for (i = 0; i < fc->outputNum; ++i) {
        fc->outputData[i] = 0;
        for (j = 0; j < fc->inputNum; ++j) {
            fc->outputData[i] += fc->inputData[j] * fc->weightData[i][j];
        }
    }
}

float ****malloc_4D(int n, int c, int i, int j)
{
    int q, w, e; 
    float ****a = (float ****) malloc(sizeof(float ***) * n);
    for (q = 0; q < n; ++q) {
        a[q] = (float ***) malloc(sizeof(float **) * c);
        for (w = 0; w < c; ++w) {
            a[q][w] = (float **) malloc(sizeof(float *) * i);
            for (e = 0; e < i; ++e) {
                a[q][w][e] = (float *) malloc(sizeof(float) * j);
            }
        }
    }
    return a; 
}

float ***malloc_3D(int c, int i, int j)
{
    int q, w;
    float ***a = (float ***) malloc(sizeof(float **) * c);
    for (q = 0; q < c; ++q) {
        a[q] = (float **) malloc(sizeof(float *) * i);
        for (w = 0; w < i; ++w) {
            a[q][w] = (float *) malloc(sizeof(float) * j);
        }
    }
    return a;
}

float **malloc_2D(int m, int n)
{
    int w;
    float **a = (float **) malloc(sizeof(float *) * m);
    for (w = 0; w < m; ++w) {
        a[w] = (float *) malloc(sizeof(float) * n);
    }
    return a;
}