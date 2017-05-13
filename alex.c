#include "cnn.h"
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int l, c, i, j;
    
    convLayer *cl_1 = (convLayer *) malloc(sizeof(convLayer)); //conv1
    int cl_1_outsize = cl_init(cl_1, 227, 11, 3, 96, 0, 4); //输出特征图的大小

    cl_1->inputData = malloc_3D(cl_1->inChannels, cl_1->h1, cl_1->w1);  //分配3D数组空间
    //动态分配4D数组：cl_1卷积核
    //cl_1->filters : sizeof(float) * cl_1->numOutput * cl_1->inChannels * cl_1->kernelSize * cl_1->kernelSize
    cl_1->filters = malloc_4D(cl_1->numOutput, cl_1->inChannels, cl_1->kernelSize, cl_1->kernelSize);
    cl_1->biasData = (float *) malloc(sizeof(float) * cl_1->numOutput); //偏置
    //动态分配3D数组：cl_1特征映射
    //cl_1->featureMap = (float ***) malloc(sizeof(float) * cl_1->numOutput * cl_1_outsize * cl_1_outsize);
    cl_1->featureMap = malloc_3D(cl_1->numOutput, cl_1_outsize, cl_1_outsize);

    //读取输入图片
    FILE *fp_pic = fopen("/home/xi/pic.txt", "r");
    for (c = 0; c < cl_1->inChannels; ++c) {
        for (i = 0; i < cl_1->h1; ++i) {
            for (j = 0; j < cl_1->w1; ++j) {
                fscanf(fp_pic, "%f", &cl_1->inputData[c][i][j]);
            }
        }
    }
    fclose(fp_pic);
    
    //读取权重
    FILE *fp_conv1_w = fopen("/home/xi/faceNet_params/conv1_weight.txt", "r");
    for (l = 0; l < cl_1->numOutput; ++l) {
        for (c = 0; c < cl_1->inChannels; ++c) {
            for (i = 0; i < cl_1->kernelSize; ++i) {
                for (j = 0; j < cl_1->kernelSize; ++j) {
                    fscanf(fp_conv1_w, "%f", &cl_1->filters[l][c][i][j]);
                }
            }
        }
    }
    fclose(fp_conv1_w);
    //读取偏置
    FILE *fp_conv1_b = fopen("/home/xi/faceNet_params/conv1_bias.txt", "r");
    for (i = 0; i < cl_1->numOutput; ++i) {
        fscanf(fp_conv1_b, "%f", &cl_1->biasData[i]);
    }
    fclose(fp_conv1_b);

    conv3D(cl_1);

    free(cl_1->inputData);
    free(cl_1->filters);
    free(cl_1->biasData);

    poolLayer *pl_1 = (poolLayer *) malloc(sizeof(poolLayer)); //pooling Layer 1
    int pl_1_outsize = pl_init(pl_1, cl_1->featureMap, cl_1_outsize, 3, cl_1->numOutput, 2);

    //动态分配3D数组：pl_1池化后数据
    //pl_1->pooledData = (float ***) malloc(sizeof(float) * pl_1->inChannels * pl_1_outsize * pl_1_outsize);
    
    pl_1->pooledData = malloc_3D(pl_1->inChannels, pl_1_outsize, pl_1_outsize);

    maxPooling(pl_1);

    free(pl_1->inputData);

    convLayer *cl_2 = (convLayer *) malloc(sizeof(convLayer)); //conv2

    int cl_2_outsize = cl_init(cl_2, pl_1_outsize, 5, pl_1->inChannels, 256, 2, 1); //输出特征图的大小

    //为了方便padding，重新分配3D数组，先全部填充为0，再填入上一层得到的结果，填充之后比原来大2*padding维
    //cl_2->inputData = (float ***) malloc(sizeof(float) * cl_2->inChannels * (cl_2->w1 + 2 * cl_2->padding) * (cl_2->h1 + 2 * cl_2->padding));

    cl_2->inputData = malloc_3D(cl_2->inChannels, cl_2->h1 + 2 * cl_2->padding, cl_2->w1 + 2 * cl_2->padding);

    //动态分配4D数组:cl_2卷积核
    //cl_2->filters: sizeof(float) * cl_2->numOutput * cl_2->inChannels * cl_2->kernelSize * cl_2->kernelSize

    cl_2->filters = malloc_4D(cl_2->numOutput, cl_2->inChannels, cl_2->kernelSize, cl_2->kernelSize);

    cl_2->biasData = (float *) malloc(sizeof(float) * cl_2->numOutput); //cl_2的偏置
    //分配3D数组：cl_2特征映射

    cl_2->featureMap = malloc_3D(cl_2->numOutput, cl_2_outsize, cl_2_outsize);
    //零填充输入特征图
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
    //读取权重
    FILE *fp_conv2_w = fopen("/home/xi/faceNet_params/conv2_weight.txt", "r");
    for (l = 0; l < cl_2->numOutput; ++l) {
        for (c = 0; c < cl_2->inChannels; ++c) {
            for (i = 0; i < cl_2->kernelSize; ++i) {
                for (j = 0; j < cl_2->kernelSize; ++j) {
                    fscanf(fp_conv2_w, "%f", &cl_2->filters[l][c][i][j]);
                }
            }
        }
    }
    fclose(fp_conv2_w);
    //读取偏置
    FILE *fp_conv2_b = fopen("/home/xi/faceNet_params/conv2_bias.txt", "r");
    for (i = 0; i < cl_2->numOutput; ++i) {
        fscanf(fp_conv2_b, "%f", &cl_2->biasData[i]);
    }
    fclose(fp_conv2_b);
    

    conv3D(cl_2);

    free(cl_2->inputData);
    free(cl_2->filters);
    free(cl_2->biasData);

    poolLayer *pl_2 = (poolLayer *) malloc(sizeof(poolLayer)); //pooling layer 2

    int pl_2_outsize = pl_init(pl_2, cl_2->featureMap, cl_2_outsize, 3, cl_2->numOutput, 2);

    //动态分配3维数组:pl_2池化后数据
    //pl_2->pooledData : sizeof(float) * pl_2->inChannels * pl_2_outsize * pl_2_outsize

    pl_2->pooledData = malloc_3D(pl_2->inChannels, pl_2_outsize, pl_2_outsize);


    maxPooling(pl_2);

    free(pl_2->inputData);

    convLayer *cl_3 = (convLayer *) malloc(sizeof(convLayer)); //conv3

    int cl_3_outsize = cl_init(cl_3, pl_2_outsize, 3, pl_2->inChannels, 384, 1, 1);

    //padding操作同上，动态分配3D数组：cl_3输入数据
    //cl_3->inputData : sizeof(float) * cl_3->inChannels * (cl_3->w1 + cl_3->padding * 2) * (cl_3->h1 + cl_3->padding * 2)

    cl_3->inputData = malloc_3D(cl_3->inChannels, cl_3->h1 + 2 * cl_3->padding, cl_3->w1 + 2 * cl_3->padding);
    //动态分配4D数组:cl_3卷积核
    //cl_3->filters : sizeof(float) * cl_3->numOutput * cl_3->inChannels * cl_3->kernelSize * cl_3->kernelSize
    cl_3->filters = malloc_4D(cl_3->numOutput, cl_3->inChannels, cl_3->kernelSize, cl_3->kernelSize);
    cl_3->biasData = (float *) malloc(sizeof(float) * cl_3->numOutput);   //为偏置分配空间
    //动态分配3D数组：cl_3特征映射
    //cl_3->featureMap = (float ***) malloc(sizeof(float) * cl_3->numOutput * cl_3_outsize * cl_3_outsize);

    cl_3->featureMap = malloc_3D(cl_3->numOutput, cl_3_outsize, cl_3_outsize);

    //零填充特征图
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
    
    //读入权重
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
    fclose(fp_conv3_w);

    //读入偏置
    FILE *fp_conv3_b = fopen("/home/xi/faceNet_params/conv3_bias.txt", "r");
    for (i = 0; i < cl_3->numOutput; ++i) {
        fscanf(fp_conv3_b, "%f", &cl_3->biasData[i]);
    }
    fclose(fp_conv3_b);

    conv3D(cl_3);

    free(cl_3->inputData);
    free(cl_3->filters);
    free(cl_3->biasData);

    convLayer *cl_4 = (convLayer *) malloc(sizeof(convLayer)); //conv4

    int cl_4_outsize = cl_init(cl_4, cl_3_outsize, 3, cl_3->numOutput, 384, 1, 1); //输出特征图的大小

    //动态分配3D数组：cl_4输入数据，需要padding
    //cl_4->inputData = (float ***) malloc(sizeof(float) * cl_4->inChannels * (cl_4->w1 + cl_4->padding * 2) + (cl_4->h1 + cl_4->padding * 2));

    cl_4->inputData = malloc_3D(cl_4->inChannels, cl_4->h1 + 2 * cl_4->padding, cl_4->w1 + 2 * cl_4->padding);
    //动态分配4D数组：cl_4卷积核
    //cl_4->filters : sizeof(float) * cl_4->numOutput * cl_4->inChannels * cl_4->kernelSize * cl_4->kernelSize
 
    cl_4->filters = malloc_4D(cl_4->numOutput, cl_4->inChannels, cl_4->kernelSize, cl_4->kernelSize);
    cl_4->biasData = (float *) malloc(sizeof(float) * cl_4->numOutput); //分配偏置空间
    //动态分配3D数组：cl_4特征映射
    //cl_4->featureMap : sizeof(float) * cl_4->numOutput * cl_4_outsize * cl_4_outsize

    cl_4->featureMap = malloc_3D(cl_4->numOutput, cl_4_outsize, cl_4_outsize);

    //零填充输入矩阵
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
    
    //读入权重
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
    fclose(fp_conv4_w);

    //读入偏置
    FILE *fp_conv4_b = fopen("/home/xi/faceNet_params/conv4_bias.txt", "r");
    for (i = 0; i < cl_4->numOutput; ++i) {
        fscanf(fp_conv4_b, "%f", &cl_4->biasData[i]);
    }
    fclose(fp_conv4_b);

    conv3D(cl_4);

    free(cl_4->inputData);
    free(cl_4->filters);
    free(cl_4->biasData);


    convLayer *cl_5 = (convLayer *) malloc(sizeof(convLayer)); //conv5

    int cl_5_outsize = cl_init(cl_5, cl_4_outsize, 3, cl_4->numOutput, 256, 1, 1);

    //动态分配3D数组：cl_5输入数据，需要padding
    //cl_5->inputData = (float ***) malloc(sizeof(float) * cl_5->inChannels * (cl_5->w1 + cl_5->padding * 2) * (cl_5->h1 + 2 * cl_5->padding));

    cl_5->inputData = malloc_3D(cl_5->inChannels, cl_5->h1 + cl_5->padding * 2, cl_5->w1 + 2 * cl_5->padding);

    //动态分配4D数组：cl_5卷积核
    //cl_5->filters = (float ****) malloc(sizeof(float) * cl_5->numOutput * cl_5->inChannels * cl_5->kernelSize * cl_5->kernelSize);

    cl_5->filters = malloc_4D(cl_5->numOutput, cl_5->inChannels, cl_5->kernelSize, cl_5->kernelSize);
    cl_5->biasData = (float *) malloc(sizeof(float) * cl_5->numOutput);
    //动态分配3D数组：cl_5特征映射
    //cl_5->featureMap = (float ***) malloc(sizeof(float) * cl_5->numOutput * cl_5_outsize * cl_5_outsize);

    cl_5->featureMap = malloc_3D(cl_5->numOutput, cl_5_outsize, cl_5_outsize);

    //零填充输入数据
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
    //读入权重
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
    fclose(fp_conv5_w);
    
    //读入偏置
    FILE *fp_conv5_b = fopen("/home/xi/faceNet_params/conv5_bias.txt", "r");
    for (i = 0; i < cl_5->numOutput; ++i) {
        fscanf(fp_conv5_b, "%f", &cl_5->biasData[i]);
    }
    fclose(fp_conv5_b);

    conv3D(cl_5);

    free(cl_5->inputData);
    free(cl_5->filters);
    free(cl_5->biasData);

    poolLayer *pl_5 = (poolLayer *) malloc(sizeof(poolLayer)); //pooling layer 5

    int pl_5_outsize = pl_init(pl_5, cl_5->featureMap, cl_5_outsize, 3, cl_5->numOutput, 2); //池化后数据的大小

    //动态分配3D数组：pl_5池化后数据
    //pl_5->pooledData : sizeof(float) * pl_5->inChannels * pl_5_outsize * pl_5_outsize
    pl_5->pooledData = malloc_3D(pl_5->inChannels, pl_5_outsize, pl_5_outsize);

    maxPooling(pl_5);

    free(pl_5->inputData);

    fcLayer *fc_6 = (fcLayer *) malloc(sizeof(fcLayer)); //full connected layer 6，由于本层上一层是池化层，故采用函数pooled2fc计算

    fc_6->outputNum = 4096;

    //分配池化层与全连接层之间的4D权重矩阵：
    //float ****pooled2fcWeights : sizeof(float) * fc_6->outputNum * pl_5->inChannels * pl_5_outsize * pl_5_outsize
    /*float ****pooled2fcWeights = (float ****) malloc(sizeof(float***) * fc_6->outputNum);
    for (l = 0; l < fc_6->outputNum; ++l) {
        pooled2fcWeights[l] = (float ***) malloc(sizeof(float**) * pl_5->inChannels);
        for (c = 0; c < pl_5->inChannels; ++c) {
            pooled2fcWeights[l][c] = (float **) malloc(sizeof(float*) * pl_5_outsize);
            for (i = 0; i < pl_5_outsize; ++i) {
                pooled2fcWeights[l][c][i] = (float *) malloc(sizeof(float) * pl_5_outsize);
            }
        }
    }*/
    float ****pooled2fcWeights = malloc_4D(fc_6->outputNum, pl_5->inChannels, pl_5_outsize, pl_5_outsize);
    fc_6->biasData = (float *) malloc(sizeof(float) * fc_6->outputNum); //分配偏置空间
    fc_6->outputData = (float *) malloc(sizeof(float) * fc_6->outputNum); //分配输出数据空间

    FILE *fp_fc6_w = fopen("/home/xi/faceNet_params/fc6_weight.txt", "r");
    for (l = 0; l < fc_6->outputNum; ++l) {
        for (c = 0; c < pl_5->inChannels; ++c) {
            for (i = 0; i < pl_5_outsize; ++i) {
                for (j = 0; j < pl_5_outsize; ++j) {
                    fscanf(fp_fc6_w, "%f", &pooled2fcWeights[l][c][i][j]);
                }
            }
        }
    }
    fclose(fp_fc6_w);

    FILE *fp_fc6_b = fopen("/home/xi/faceNet_params/fc6_bias.txt", "r");
    for (i = 0; i < fc_6->outputNum; ++i) {
        fscanf(fp_fc6_b, "%f", &fc_6->biasData[i]);
    }
    fclose(fp_fc6_b);

    pooled2fc(*pl_5, pooled2fcWeights, fc_6);

    free(pl_5->pooledData);
    free(pooled2fcWeights);
    free(fc_6->biasData);

    fcLayer *fc_7 = (fcLayer *) malloc(sizeof(fcLayer)); //full connected layer 7
    fc_7->inputNum = fc_6->outputNum;  
    fc_7->outputNum = 4096; 
    fc_7->inputData = fc_6->outputData;

    //分配fc_7的2D权重矩阵
    //fc_7->weightData : sizeof(float) * fc_7->inputNum * fc_7->outputNum
    /*fc_7->weightData = (float **) malloc(sizeof(float*) * fc_7->outputNum);
    for (i = 0; i < fc_7->outputNum; ++i) {
        fc_7->weightData[i] = (float *) malloc(sizeof(float) * fc_7->inputNum);
    }*/
    fc_7->weightData = malloc_2D(fc_7->outputNum, fc_7->inputNum);
    fc_7->biasData = (float *) malloc(sizeof(float) * fc_7->outputNum);   //分配偏置数组
    fc_7->outputData = (float *) malloc(sizeof(float) * fc_7->outputNum);  //分配输出数据

    FILE *fp_fc7_w = fopen("/home/xi/faceNet_params/fc7_weight.txt", "r");
    for (i = 0; i < fc_7->outputNum; ++i) {
        for (j = 0; j < fc_7->inputNum; ++j) {
            fscanf(fp_fc7_w, "%f", &fc_7->weightData[i][j]);
        }
    }
    fclose(fp_fc7_w);

    FILE *fp_fc7_b = fopen("/home/xi/faceNet_params/fc7_bias.txt", "r");
    for (i = 0; i < fc_7->outputNum; ++i) {
        fscanf(fp_fc7_b, "%f", &fc_7->biasData[i]);
    }
    fclose(fp_fc7_b);

    fc2fc(fc_7);

    free(fc_7->inputData);
    free(fc_7->weightData);
    free(fc_7->biasData);

    fcLayer *fc_8 = (fcLayer *) malloc(sizeof(fcLayer)); //full connected layer 8
    fc_8->inputNum = fc_7->outputNum;  
    fc_8->outputNum = 100; 

    //分配fc_7的2D权重矩阵
    //fc_8->weightData = (float **) malloc(sizeof(float) * fc_8->inputNum * fc_8->outputNum); 
    /*fc_8->weightData = (float **) malloc(sizeof(float*) * fc_8->outputNum);
    for (i = 0; i < fc_8->outputNum; ++i) {
        fc_8->weightData[i] = (float *) malloc(sizeof(float) * fc_8->inputNum);
    }*/
    fc_8->weightData = malloc_2D(fc_8->outputNum, fc_8->inputNum);
    fc_8->biasData = (float *) malloc(sizeof(float) * fc_8->outputNum);   
    fc_8->inputData = fc_7->outputData;
    fc_8->outputData = (float *) malloc(sizeof(float) * fc_8->outputNum);

    FILE *fp_fc8_w = fopen("/home/xi/faceNet_params/fc8_weight.txt", "r");
    for (i = 0; i < fc_8->outputNum; ++i) {
        for (j = 0; j < fc_8->inputNum; ++j) {
            fscanf(fp_fc8_w, "%f", &fc_8->weightData[i][j]);
        }
    }
    fclose(fp_fc8_w);

    FILE *fp_fc8_b = fopen("/home/xi/faceNet_params/fc8_bias.txt", "r");
    for (i = 0; i < fc_8->outputNum; ++i) {
        fscanf(fp_fc8_b, "%f", &fc_8->biasData[i]);
    }
    fclose(fp_fc8_b);

    fc2fc(fc_8);

    free(fc_8->inputData);
    free(fc_8->weightData);
    free(fc_8->biasData);
    
    for (i = 0; i < fc_8->outputNum; ++i) {
        printf("%f ", fc_8->outputData[i]);
    }

    return 0;
}
