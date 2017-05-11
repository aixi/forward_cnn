#include "cnn.h"
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    FILE *fp_pic = fopen("/home/xi/pic.txt", "r");
    FILE *fp_conv1_w = fopen("/home/xi/faceNet_params/conv1_weight.txt", "r");
    FILE *fp_conv1_b = fopen("/home/xi/faceNet_params/conv1_bias.txt", "r");
    int l, c, i, j;
    convLayer *cl_1 = (convLayer *) malloc(sizeof(convLayer)); //conv1
    cl_1->w1 = cl_1->h1 = 227;
    cl_1->inChannels = 3;
    cl_1->kernelSize = 11;
    cl_1->numOutput = 96;
    cl_1->padding = 0;
    cl_1->stride = 4;
    //动态分配3D数组：输入图片
    cl_1->inputData = (float ***) malloc(sizeof(float**) * cl_1->inChannels); 
    for (c = 0; c < cl_1->inChannels; ++c) {
        cl_1->inputData[c] = (float **) malloc(sizeof(float*) * cl_1->h1);
        for (i = 0; i < cl_1->h1; ++i) {
            cl_1->inputData[c][i] = (float *) malloc(sizeof(float) * cl_1->w1);
        }
    }
    for (c = 0; c < cl_1->inChannels; ++c) {
        for (i = 0; i < cl_1->h1; ++i) {
            for (j = 0; j < cl_1->w1; ++j) {
                fscanf(fp_pic, "%f", &cl_1->inputData[c][i][j]);
            }
        }
    }
    int cl_1_outsize = (cl_1->w1 - cl_1->kernelSize + cl_1->padding * 2) / cl_1->stride + 1;
    //动态分配4D数组：cl_1卷积核
    //cl_1->filters : sizeof(float) * cl_1->numOutput * cl_1->inChannels * cl_1->kernelSize * cl_1->kernelSize
    cl_1->filters = (float ****) malloc(sizeof(float***) * cl_1->numOutput);
    for (l = 0; l < cl_1->numOutput; ++l) {
        cl_1->filters[l] = (float ***) malloc(sizeof(float**) * cl_1->inChannels);
        for (c = 0; c < cl_1->inChannels; ++c) {
            cl_1->filters[l][c] = (float **) malloc(sizeof(float*) * cl_1->kernelSize);
            for (i = 0; i < cl_1->kernelSize; ++i) {
                cl_1->filters[l][c][i] = (float *) malloc(sizeof(float) * cl_1->kernelSize);
            }
        }
    }
    for (l = 0; l < cl_1->numOutput; ++l) {
        for (c = 0; c < cl_1->inChannels; ++c) {
            for (i = 0; i < cl_1->kernelSize; ++i) {
                for (j = 0; j < cl_1->kernelSize; ++j) {
                    fscanf(fp_conv1_w, "%f", &cl_1->filters[l][c][i][j]);
                }
            }
        }
    }
    cl_1->biasData = (float *) malloc(sizeof(float) * cl_1->numOutput);
    for (i = 0; i < cl_1->numOutput; ++i) {
        fscanf(fp_conv1_b, "%f", &cl_1->biasData[i]);
    }
    //动态分配3D数组：cl_1特征映射
    //cl_1->featureMap = (float ***) malloc(sizeof(float) * cl_1->numOutput * cl_1_outsize * cl_1_outsize);
    cl_1->featureMap = (float ***) malloc(sizeof(float**) * cl_1->numOutput);
    for (c = 0; c < cl_1->numOutput; ++c) {
        cl_1->featureMap[c] = (float **) malloc(sizeof(float*) * cl_1_outsize);
        for (i = 0; i < cl_1_outsize; ++i) {
            cl_1->featureMap[c][i] = (float *) malloc(sizeof(float) * cl_1_outsize);
        }
    }


    conv3D(cl_1);
    /*for (c = 0; c < cl_1->numOutput; ++c) {
        for (i = 0; i < cl_1_outsize; ++i) {
            for (j = 0; j < cl_1_outsize; ++j) {
                printf("%f ", cl_1->featureMap[c][i][j]);
            }
        }
    }*/

    poolLayer *pl_1 = (poolLayer *) malloc(sizeof(poolLayer)); //pooling Layer 1
    pl_1->w1 = pl_1->h1 = cl_1_outsize;
    pl_1->kernelSize = 3;
    pl_1->stride = 2;
    pl_1->inChannels = cl_1->numOutput;
    pl_1->inputData = cl_1->featureMap;
    int pl_1_outsize = (pl_1->w1 - pl_1->kernelSize) / pl_1->stride + 1;
    //动态分配3D数组：pl_1池化后数据
    //pl_1->pooledData = (float ***) malloc(sizeof(float) * pl_1->inChannels * pl_1_outsize * pl_1_outsize);
    pl_1->pooledData = (float ***) malloc(sizeof(float**) * pl_1->inChannels);
    for (c = 0; c < pl_1->inChannels; ++c) {
        pl_1->pooledData[c] = (float **) malloc(sizeof(float*) * pl_1_outsize);
        for (i = 0; i < pl_1_outsize; ++i) {
            pl_1->pooledData[c][i] = (float *) malloc(sizeof(float) * pl_1_outsize);
        }
    }

    //printf("\n\n\n\n\n");
    maxPooling(pl_1);

    /*for (c = 0; c < pl_1->inChannels; ++c) {
        for (i = 0; i < pl_1_outsize; ++i) {
            for (j = 0; j < pl_1_outsize; ++j) {
                printf("%f ", pl_1->pooledData[c][i][j]);
            }
        }
    }*/


    FILE *fp_conv2_w = fopen("/home/xi/faceNet_params/conv2_weight.txt", "r");
    FILE *fp_conv2_b = fopen("/home/xi/faceNet_params/conv2_bias.txt", "r");
    convLayer *cl_2 = (convLayer *) malloc(sizeof(convLayer)); //conv2
    cl_2->w1 = cl_2->h1 = pl_1_outsize;
    cl_2->stride = 1;
    cl_2->kernelSize = 5;
    cl_2->numOutput = 256;
    cl_2->inChannels = pl_1->inChannels;
    cl_2->padding = 2;
    //为了方便padding，重新分配3D数组，先全部填充为0，再填入上一层得到的结果，填充之后比原来大2*padding维
    //cl_2->inputData = (float ***) malloc(sizeof(float) * cl_2->inChannels * (cl_2->w1 + 2 * cl_2->padding) * (cl_2->h1 + 2 * cl_2->padding));
    cl_2->inputData = (float ***) malloc(sizeof(float**) * cl_2->inChannels);
    for (c = 0; c < cl_2->inChannels; ++c) {
        cl_2->inputData[c] = (float **) malloc(sizeof(float*) * (cl_2->h1 + 2 * cl_2->padding));
        for (i = 0; i < cl_2->h1 + 2 * cl_2->padding; ++i) {
            cl_2->inputData[c][i] = (float *) malloc(sizeof(float) * (cl_2->w1 + 2 * cl_2->padding));
        }
    }
    for (c = 0; c < cl_2->inChannels; ++c) {
        for (i = 0; i < cl_2->h1 + 2 * cl_2->padding; ++i) {
            for (j = 0; j < cl_2->w1 + 2 * cl_2->padding; ++j) {
                cl_2->inputData[c][i][j] = 0;
            }
        }
    }
    for (c = 0; c < cl_2->inChannels; ++c) {
        for (i = cl_2->padding; i < cl_2->padding + pl_1_outsize; ++i) {
            for (j = cl_2->padding; j < cl_2->padding + pl_1_outsize; ++j) {
                cl_2->inputData[c][i][j] = pl_1->pooledData[c][i-cl_2->padding][j-cl_2->padding];
            }
        }
    }
    int cl_2_outsize = (cl_2->w1 - cl_2->kernelSize + cl_2->padding * 2) / cl_2->stride + 1;
    //动态分配4D数组:cl_2卷积核
    //cl_2->filters: sizeof(float) * cl_2->numOutput * cl_2->inChannels * cl_2->kernelSize * cl_2->kernelSize
    cl_2->filters = (float ****) malloc(sizeof(float***) * cl_2->numOutput);
    for (l = 0; l < cl_2->numOutput; ++l) {
        cl_2->filters[l] = (float ***) malloc(sizeof(float**) * cl_2->inChannels);
        for (c = 0; c < cl_2->inChannels; ++c) {
            cl_2->filters[l][c] = (float **) malloc(sizeof(float*) * cl_2->kernelSize);
            for (i = 0; i < cl_2->kernelSize; ++i) {
                cl_2->filters[l][c][i] = (float *) malloc(sizeof(float) * cl_2->kernelSize);
            }
        }
    }
    for (l = 0; l < cl_2->numOutput; ++l) {
        for (c = 0; c < cl_2->inChannels; ++c) {
            for (i = 0; i < cl_2->kernelSize; ++i) {
                for (j = 0; j < cl_2->kernelSize; ++j) {
                    fscanf(fp_conv2_w, "%f", &cl_2->filters[l][c][i][j]);
                }
            }
        }
    }
    /*for (l = 0; l < cl_2->numOutput; ++l) {
        for (c = 0; c < cl_2->inChannels; ++c) {
            for (i = 0; i < cl_2->kernelSize; ++i) {
                for (j = 0; j < cl_2->kernelSize; ++j) {
                    printf("%f ", cl_2->filters[l][c][i][j]);
                }
            }
        }
    }*/
    cl_2->biasData = (float *) malloc(sizeof(float) * cl_2->numOutput);
    for (i = 0; i < cl_2->numOutput; ++i) {
        fscanf(fp_conv2_b, "%f", &cl_2->biasData[i]);
    }
    
    //分配3D数组：cl_2特征映射
    cl_2->featureMap = (float ***) malloc(sizeof(float**) * cl_2->numOutput);
    for (c = 0; c < cl_2->numOutput; ++c) {
        cl_2->featureMap[c] = (float **) malloc(sizeof(float*) * cl_2_outsize);
        for (i = 0; i < cl_2_outsize; ++i) {
            cl_2->featureMap[c][i] = (float *) malloc(sizeof(float) * cl_2_outsize);
        }
    }

    conv3D(cl_2);

    /*for (c = 0; c < cl_2->numOutput; ++c) {
        for (i = 0; i < cl_2_outsize; ++i) {
            for (j = 0; j < cl_2_outsize; ++j) {
                printf("%f ", cl_2->featureMap[c][i][j]);
            }
        }
    }*/

    poolLayer *pl_2 = (poolLayer *) malloc(sizeof(poolLayer)); //pooling layer 2
    pl_2->w1 = pl_2->h1 = cl_2_outsize;
    pl_2->kernelSize = 3;
    pl_2->stride = 2;
    pl_2->inChannels = cl_2->numOutput;
    pl_2->inputData = cl_2->featureMap;
    int pl_2_outsize = (pl_2->w1 - pl_2->kernelSize) / pl_2->stride + 1;
    //动态分配3维数组:pl_2池化后数据
    //pl_2->pooledData : sizeof(float) * pl_2->inChannels * pl_2_outsize * pl_2_outsize
    pl_2->pooledData = (float ***) malloc(sizeof(float**) * pl_2->inChannels);
    for (c = 0; c < pl_2->inChannels; ++c) {
        pl_2->pooledData[c] = (float **) malloc(sizeof(float*) * pl_2_outsize);
        for (i = 0; i < pl_2_outsize; ++i) {
            pl_2->pooledData[c][i] = (float *) malloc(sizeof(float) * pl_2_outsize);
        }
    }


    maxPooling(pl_2);

    /*for (c = 0; c < pl_2->inChannels; ++c) {
        for (i = 0; i < pl_2_outsize; ++i) {
            for (j = 0; j < pl_2_outsize; ++j) {
                printf("%f ", pl_2->pooledData[c][i][j]);
            }
        }
    }*/

    convLayer *cl_3 = (convLayer *) malloc(sizeof(convLayer)); //conv3
    cl_3->w1 = cl_3->h1 = pl_2_outsize;
    cl_3->inChannels = pl_2->inChannels;
    cl_3->stride = 1;
    cl_3->kernelSize = 3;
    cl_3->numOutput = 384;
    cl_3->padding = 1;
    int cl_3_outsize = (cl_3->w1 - cl_3->kernelSize + 2 * cl_3->padding) / cl_3->stride + 1;
    //padding操作同上，动态分配3D数组：cl_3输入数据
    //cl_3->inputData : sizeof(float) * cl_3->inChannels * (cl_3->w1 + cl_3->padding * 2) * (cl_3->h1 + cl_3->padding * 2)
    cl_3->inputData = (float ***) malloc(sizeof(float**) * cl_3->inChannels);
    for (c = 0; c < cl_3->inChannels; ++c) {
        cl_3->inputData[c] = (float **) malloc(sizeof(float*) * (cl_3->h1 + 2 * cl_3->padding));
        for (i = 0; i < cl_3->h1 + 2 * cl_3->padding; ++i) {
            cl_3->inputData[c][i] = (float *) malloc(sizeof(float) * (cl_3->w1 + 2 * cl_3->padding));
        }
    }
    for (c = 0; c < cl_3->inChannels; ++c) {
        for (i = 0; i < cl_3->h1 + 2 * cl_3->padding; ++i) {
            for (j = 0; j < cl_3->w1 + 2 * cl_3->padding; ++j) {
                cl_3->inputData[c][i][j] = 0;
            }
        }
    }
    for (c = 0; c < cl_3->inChannels; ++c) {
        for (i = cl_3->padding; i < cl_3->padding + pl_2_outsize; ++i) {
            for (j = cl_3->padding; j < cl_3->padding + pl_2_outsize; ++j) {
                cl_3->inputData[c][i][j] = pl_2->pooledData[c][i-cl_3->padding][j-cl_3->padding];
            }
        }
    }
    //动态分配4D数组:cl_3卷积核
    //cl_3->filters : sizeof(float) * cl_3->numOutput * cl_3->inChannels * cl_3->kernelSize * cl_3->kernelSize
    cl_3->filters = (float ****) malloc(sizeof(float***) * cl_3->numOutput);
    for (l = 0; l < cl_3->numOutput; ++l) {
        cl_3->filters[l] = (float ***) malloc(sizeof(float**) * cl_3->inChannels);
        for (c = 0; c < cl_3->inChannels; ++c) {
            cl_3->filters[l][c] = (float **) malloc(sizeof(float*) * cl_3->kernelSize);
            for (i = 0; i < cl_3->kernelSize; ++i) {
                cl_3->filters[l][c][i] = (float *) malloc(sizeof(float) * cl_3->kernelSize);
            }
        }
    }
    cl_3->biasData = (float *) malloc(sizeof(float) * cl_3->numOutput);
    //动态分配3D数组：cl_3特征映射
    //cl_3->featureMap = (float ***) malloc(sizeof(float) * cl_3->numOutput * cl_3_outsize * cl_3_outsize);
    cl_3->featureMap = (float ***) malloc(sizeof(float**) * cl_3->numOutput);
    for (c = 0; c < cl_3->numOutput; ++c) {
        cl_3->featureMap[c] = (float **) malloc(sizeof(float*) * cl_3_outsize);
        for (i = 0; i < cl_3_outsize; ++i) {
            cl_3->featureMap[c][i] = (float *) malloc(sizeof(float) * cl_3_outsize);
        }
    }
    FILE *fp_conv3_w = fopen("/home/xi/faceNet_params/conv3_weight.txt", "r");
    for (l = 0; l < cl_3->numOutput; ++l) {
        for (c = 0; c < cl_3->inChannels; ++c) {
            for (i = 0; i < cl_3->kernelSize; ++i) {
                for (j = 0; j < cl_3->kernelSize; ++j) {
                    //cl_3->filters[l][c][i][j] = 0;
                    fscanf(fp_conv3_w, "%f", &cl_3->filters[l][c][i][j]);
                }
            }
        }
    }
    FILE *fp_conv3_b = fopen("/home/xi/faceNet_params/conv3_bias.txt", "r");
    for (i = 0; i < cl_3->numOutput; ++i) {
        fscanf(fp_conv3_b, "%f", &cl_3->biasData[i]);
    }


    conv3D(cl_3);

    /*for (c = 0; c < cl_3->numOutput; ++c) {
        for (i = 0; i < cl_3_outsize; ++i) {
            for (j = 0; j < cl_3_outsize; ++j) {
                printf("%f ", cl_3->featureMap[c][i][j]);
            }
        }
    }*/

    convLayer *cl_4 = (convLayer *) malloc(sizeof(convLayer)); //conv4
    cl_4->w1 = cl_4->h1 = cl_3_outsize;   
    cl_4->stride = 1; 
    cl_4->kernelSize = 3;     
    cl_4->inChannels = cl_3->numOutput;     
    cl_4->numOutput = 384;     
    cl_4->padding = 1;
    int cl_4_outsize = (cl_4->w1 - cl_4->kernelSize + 2 * cl_4->padding) / cl_4->stride + 1;
    //动态分配3D数组：cl_4特征映射
    //cl_4->featureMap : sizeof(float) * cl_4->numOutput * cl_4_outsize * cl_4_outsize
    cl_4->featureMap = (float ***) malloc(sizeof(float**) * cl_4->numOutput);
    for (c = 0; c < cl_4->numOutput; ++c) {
        cl_4->featureMap[c] = (float **) malloc(sizeof(float*) * cl_4_outsize);
        for (i = 0; i < cl_4_outsize; ++i) {
            cl_4->featureMap[c][i] = (float *) malloc(sizeof(float) * cl_4_outsize);
        }
    }
    //动态分配3D数组：cl_4输入数据，需要padding
    //cl_4->inputData = (float ***) malloc(sizeof(float) * cl_4->inChannels * (cl_4->w1 + cl_4->padding * 2) + (cl_4->h1 + cl_4->padding * 2));
    cl_4->inputData = (float ***) malloc(sizeof(float**) * cl_4->inChannels);
    for (c = 0; c < cl_4->inChannels; ++c) {
        cl_4->inputData[c] = (float **) malloc(sizeof(float*) * (cl_4->h1 + 2 * cl_4->padding));
        for (i = 0; i < cl_4->h1 + 2 * cl_4->padding; ++i) {
            cl_4->inputData[c][i] = (float *) malloc(sizeof(float) * (cl_4->w1 + 2 * cl_4->padding));
        }
    }
    for (c = 0; c < cl_4->inChannels; ++c) {
        for (i = 0; i < cl_4->h1 + 2 * cl_4->padding; ++i) {
            for (j = 0; j < cl_4->w1 + 2 * cl_4->padding; ++j) {
                cl_4->inputData[c][i][j] = 0;
            }
        }
    }
    for (c = 0; c < cl_4->inChannels; ++c) {
        for (i = cl_4->padding; i < cl_4->padding + cl_3_outsize; ++i) {
            for (j = cl_4->padding; j < cl_4->padding + cl_3_outsize; ++j) {
                cl_4->inputData[c][i][j] = cl_3->featureMap[c][i-cl_4->padding][j-cl_4->padding];
            }
        }
    }
    //动态分配4D数组：cl_4卷积核
    //cl_4->filters : sizeof(float) * cl_4->numOutput * cl_4->inChannels * cl_4->kernelSize * cl_4->kernelSize
    cl_4->filters = (float ****) malloc(sizeof(float***) * cl_4->numOutput);
    for (l = 0; l < cl_4->numOutput; ++l) {
        cl_4->filters[l] = (float ***) malloc(sizeof(float**) * cl_4->inChannels);
        for (c = 0; c < cl_4->inChannels; ++c) {
            cl_4->filters[l][c] = (float **) malloc(sizeof(float*) * cl_4->kernelSize);
            for (i = 0; i < cl_4->h1; ++i) {
                cl_4->filters[l][c][i] = (float *) malloc(sizeof(float) * cl_4->kernelSize);
            }
        }
    }
    FILE *fp_conv4_w = fopen("/home/xi/faceNet_params/conv4_weight.txt", "r");
    for (l = 0; l < cl_4->numOutput; ++l) {
        for (c = 0; c < cl_4->inChannels; ++c) {
            for (i = 0; i < cl_4->kernelSize; ++i) {
                for (j = 0; j < cl_4->kernelSize; ++j) {
                    fscanf(fp_conv4_w, "%f", &cl_4->filters[l][c][i][j]);
                }
            }
        }
    }
    cl_4->biasData = (float *) malloc(sizeof(float) * cl_4->numOutput);
    FILE *fp_conv4_b = fopen("/home/xi/faceNet_params/conv4_bias.txt", "r");
    for (i = 0; i < cl_4->numOutput; ++i) {
        fscanf(fp_conv4_b, "%f", &cl_4->biasData[i]);
    }


    conv3D(cl_4);

    /*for (c = 0; c < cl_4->numOutput; ++c) {
        for (i = 0; i < cl_4_outsize; ++i) {
            for (j = 0; j < cl_4_outsize; ++j) {
                printf("%f ", cl_4->featureMap[c][i][j]);
            }
        }
    }*/

    convLayer *cl_5 = (convLayer *) malloc(sizeof(convLayer)); //conv5
    cl_5->w1 = cl_5->h1 = cl_4_outsize;   
    cl_5->stride = 1; 
    cl_5->kernelSize = 3;     
    cl_5->inChannels = cl_4->numOutput;     
    cl_5->numOutput = 256;     
    cl_5->padding = 1;
    //动态分配4D数组：cl_5卷积核
    //cl_5->filters = (float ****) malloc(sizeof(float) * cl_5->numOutput * cl_5->inChannels * cl_5->w1 * cl_5->h1);
    cl_5->filters = (float ****) malloc(sizeof(float***) * cl_5->numOutput);
    for (l = 0; l < cl_5->numOutput; ++l) {
        cl_5->filters[l] = (float ***) malloc(sizeof(float**) * cl_5->inChannels);
        for (c = 0; c < cl_5->inChannels; ++c) {
            cl_5->filters[l][c] = (float **) malloc(sizeof(float*) * cl_5->kernelSize);
            for (i = 0; i < cl_5->h1; ++i) {
                cl_5->filters[l][c][i] = (float *) malloc(sizeof(float) * cl_5->kernelSize);
            }
        }
    }

    int cl_5_outsize = (cl_5->w1 - cl_5->kernelSize + cl_5->padding * 2) / cl_5->stride + 1;

    FILE *fp_conv5_w = fopen("/home/xi/faceNet_params/conv5_weight.txt", "r");
    for (l = 0; l < cl_5->numOutput; ++l) {
        for (c = 0; c < cl_5->inChannels; ++c) {
            for (i = 0; i < cl_5->kernelSize; ++i) {
                for (j = 0; j < cl_5->kernelSize; ++j) {
                    fscanf(fp_conv5_w, "%f", &cl_5->filters[l][c][i][j]);
                }
            }
        }
    }

    cl_5->biasData = (float *) malloc(sizeof(float) * cl_5->numOutput);
    FILE *fp_conv5_b = fopen("/home/xi/faceNet_params/conv5_bias.txt", "r");
    for (i = 0; i < cl_5->numOutput; ++i) {
        fscanf(fp_conv5_b, "%f", &cl_5->biasData[i]);
    }

    //动态分配3D数组：cl_5输入数据，需要padding
    //cl_5->inputData = (float ***) malloc(sizeof(float) * cl_5->inChannels * (cl_5->w1 + cl_5->padding * 2) * (cl_5->h1 + 2 * cl_5->padding));
    cl_5->inputData = (float ***) malloc(sizeof(float**) * cl_5->inChannels);
    for (c = 0; c < cl_5->inChannels; ++c) {
        cl_5->inputData[c] = (float **) malloc(sizeof(float*) * (cl_5->h1 + 2 * cl_5->padding));
        for (i = 0; i < cl_5->h1 + 2 * cl_5->padding; ++i) {
            cl_5->inputData[c][i] = (float *) malloc(sizeof(float) * (cl_5->w1 + 2 * cl_5->padding));
        }
    }
    for (c = 0; c < cl_5->inChannels; ++c) {
        for (i = 0; i < cl_5->h1 + 2 * cl_5->padding; ++i) {
            for (j = 0; j < cl_5->w1 + 2 * cl_5->padding; ++j) {
                cl_5->inputData[c][i][j] = 0;
            }
        }
    }
    for (c = 0; c < cl_5->inChannels; ++c) {
        for (i = cl_5->padding; i < cl_5->padding + cl_4_outsize; ++i) {
            for (j = cl_5->padding; j < cl_5->padding + cl_4_outsize; ++j) {
                cl_5->inputData[c][i][j] = cl_4->featureMap[c][i-cl_5->padding][j-cl_5->padding];
            }
        }
    }

    //动态分配3D数组：cl_5特征映射
    //cl_5->featureMap = (float ***) malloc(sizeof(float) * cl_5->numOutput * cl_5_outsize * cl_5_outsize);
    cl_5->featureMap = (float ***) malloc(sizeof(float**) * cl_5->numOutput);
    for (c = 0; c < cl_5->numOutput; ++c) {
        cl_5->featureMap[c] = (float **) malloc(sizeof(float*) * cl_5_outsize);
        for (i = 0; i < cl_5_outsize; ++i) {
            cl_5->featureMap[c][i] = (float *) malloc(sizeof(float) * cl_5_outsize);
        }
    }

    conv3D(cl_5);

    /*for (c = 0; c < cl_5->numOutput; ++c) {
        for (i = 0; i < cl_5_outsize; ++i) {
            for (j = 0; j < cl_5_outsize; ++j) {
                printf("%f ", cl_5->featureMap[c][i][j]);
            }
        }
    }*/

    poolLayer *pl_5 = (poolLayer *) malloc(sizeof(poolLayer)); //pooling layer 5
    pl_5->w1 = pl_5->h1 = cl_5_outsize;
    pl_5->kernelSize = 3;
    pl_5->stride = 2;
    pl_5->inChannels = cl_5->numOutput;
    pl_5->inputData = cl_5->featureMap;

    int pl_5_outsize = (pl_5->w1 - pl_5->kernelSize) / pl_5->stride + 1;

    //动态分配3D数组：pl_5池化后数据
    //pl_5->pooledData : sizeof(float) * pl_5->inChannels * pl_5_outsize * pl_5_outsize
    pl_5->pooledData = (float ***) malloc(sizeof(float**) * pl_5->inChannels);
    for (c = 0; c < pl_5->inChannels; ++c) {
        pl_5->pooledData[c] = (float **) malloc(sizeof(float*) * pl_5_outsize);
        for (i = 0; i < pl_5_outsize; ++i) {
            pl_5->pooledData[c][i] = (float *) malloc(sizeof(float) * pl_5_outsize);
        }
    }

    maxPooling(pl_5);

    /*for (c = 0; c < pl_5->inChannels; ++c) {
        for (i = 0; i < pl_5_outsize; ++i) {
            for (j = 0; j < pl_5_outsize; ++j) {
                printf("%f ", pl_5->pooledData[c][i][j]);
            }
        }
    }*/

    fcLayer *fc_6 = (fcLayer *) malloc(sizeof(fcLayer)); //full connected layer 6，由于本层上一层是池化层，故采用函数pooled2fc计算
    fc_6->outputNum = 4096;
    fc_6->outputData = (float *) malloc(sizeof(float) * fc_6->outputNum);

    //分配池化层与全连接层之间的4D权重矩阵：
    //float ****pooled2fcWeights : sizeof(float) * fc_6->outputNum * pl_5->inChannels * pl_5_outsize * pl_5_outsize
    float ****pooled2fcWeights = (float ****) malloc(sizeof(float***) * fc_6->outputNum);
    for (l = 0; l < fc_6->outputNum; ++l) {
        pooled2fcWeights[l] = (float ***) malloc(sizeof(float**) * pl_5->inChannels);
        for (c = 0; c < pl_5->inChannels; ++c) {
            pooled2fcWeights[l][c] = (float **) malloc(sizeof(float*) * pl_5_outsize);
            for (i = 0; i < pl_5_outsize; ++i) {
                pooled2fcWeights[l][c][i] = (float *) malloc(sizeof(float) * pl_5_outsize);
            }
        }
    }

    //long count = 0;
    FILE *fp_fc6_w = fopen("/home/xi/faceNet_params/fc6_weight.txt", "r");
    for (l = 0; l < fc_6->outputNum; ++l) {
        for (c = 0; c < pl_5->inChannels; ++c) {
            for (i = 0; i < pl_5_outsize; ++i) {
                for (j = 0; j < pl_5_outsize; ++j) {
                    fscanf(fp_fc6_w, "%f", &pooled2fcWeights[l][c][i][j]);
                    //printf("%f ", pooled2fcWeights[l][c][i][j]);
                    //++count;
                }
            }
        }
    }

    /*for (l = 0; l < fc_6->outputNum; ++l) {
        for (c = 0; c < pl_5->inChannels; ++c) {
            for (i = 0; i < pl_5->kernelSize; ++i) {
                for (j = 0; j < pl_5->kernelSize; ++j) {
                    printf("%f ", pooled2fcWeights[l][c][i][j]);
                }
            }
        }
    }*/

    //printf("\n\n\n\n%ld\n\n\n\n", count);

    fc_6->biasData = (float *) malloc(sizeof(float) * fc_6->outputNum);
    FILE *fp_fc6_b = fopen("/home/xi/faceNet_params/fc6_bias.txt", "r");
    for (i = 0; i < fc_6->outputNum; ++i) {
        fscanf(fp_fc6_b, "%f", &fc_6->biasData[i]);
    }

    pooled2fc(*pl_5, pooled2fcWeights, fc_6);

    /*for (i = 0; i < fc_6->outputNum; ++i) {
        printf("%f ", fc_6->outputData[i]);
    }*/

    fcLayer *fc_7 = (fcLayer *) malloc(sizeof(fcLayer)); //full connected layer 7
    fc_7->inputNum = fc_6->outputNum;  
    fc_7->outputNum = 4096; 

    //分配fc_7的2D权重矩阵
    //fc_7->weightData : sizeof(float) * fc_7->inputNum * fc_7->outputNum
    fc_7->weightData = (float **) malloc(sizeof(float*) * fc_7->outputNum);
    for (i = 0; i < fc_7->outputNum; ++i) {
        fc_7->weightData[i] = (float *) malloc(sizeof(float) * fc_7->inputNum);
    }
    fc_7->biasData = (float *) malloc(sizeof(float) * fc_7->outputNum);   
    fc_7->inputData = fc_6->outputData;
    fc_7->outputData = (float *) malloc(sizeof(float) * fc_7->outputNum);

    FILE *fp_fc7_w = fopen("/home/xi/faceNet_params/fc7_weight.txt", "r");
    for (i = 0; i < fc_7->outputNum; ++i) {
        for (j = 0; j < fc_7->inputNum; ++j) {
            fscanf(fp_fc7_w, "%f", &fc_7->weightData[i][j]);
        }
    }

    FILE *fp_fc7_b = fopen("/home/xi/faceNet_params/fc7_bias.txt", "r");
    for (i = 0; i < fc_7->outputNum; ++i) {
        fscanf(fp_fc7_b, "%f", &fc_7->biasData[i]);
    }

    fc2fc(fc_7);

    /*for (i = 0; i < fc_7->outputNum; ++i) {
        printf("%f ", fc_7->outputData[i]);
    }*/

    fcLayer *fc_8 = (fcLayer *) malloc(sizeof(fcLayer)); //full connected layer 8
    fc_8->inputNum = fc_7->outputNum;  
    fc_8->outputNum = 100; 

    //分配fc_7的2D权重矩阵
    //fc_8->weightData = (float **) malloc(sizeof(float) * fc_8->inputNum * fc_8->outputNum); 
    fc_8->weightData = (float **) malloc(sizeof(float*) * fc_8->outputNum);
    for (i = 0; i < fc_8->outputNum; ++i) {
        fc_8->weightData[i] = (float *) malloc(sizeof(float) * fc_8->inputNum);
    }
    fc_8->biasData = (float *) malloc(sizeof(float) * fc_8->outputNum);   
    fc_8->inputData = fc_7->outputData;
    fc_8->outputData = (float *) malloc(sizeof(float) * fc_8->outputNum);

    FILE *fp_fc8_w = fopen("/home/xi/faceNet_params/fc8_weight.txt", "r");
    for (i = 0; i < fc_8->outputNum; ++i) {
        for (j = 0; j < fc_8->inputNum; ++j) {
            fscanf(fp_fc8_w, "%f", &fc_8->weightData[i][j]);
        }
    }

    FILE *fp_fc8_b = fopen("/home/xi/faceNet_params/fc8_bias.txt", "r");
    for (i = 0; i < fc_8->outputNum; ++i) {
        fscanf(fp_fc8_b, "%f", &fc_8->biasData[i]);
    }

    fc2fc(fc_8);

    for (i = 0; i < fc_8->outputNum; ++i) {
        printf("%f ", fc_8->outputData[i]);
    }
    
    printf("\n");

    return 0;
}

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