#include "cnn.h"
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int l, c, i, j;
    convLayer *cl_1 = (convLayer *) malloc(sizeof(convLayer));
    int cl_1_outsize = cl_init(cl_1, 227, 11, 3, 96, 0, 4);

    cl_1->inputData = malloc_3D(cl_1->inChannels, cl_1->h1, cl_1->w1);
    cl_1->filters = malloc_4D(cl_1->numOutput, cl_1->inChannels, cl_1->kernelSize, cl_1->kernelSize);
    cl_1->biasData = (float *) malloc(sizeof(float) * cl_1->numOutput); 
    cl_1->featureMap = malloc_3D(cl_1->numOutput, cl_1_outsize, cl_1_outsize);

    FILE *fp_pic = fopen("/home/xi/pic.txt", "r");
    for (c = 0; c < cl_1->inChannels; ++c) {
        for (i = 0; i < cl_1->h1; ++i) {
            for (j = 0; j < cl_1->w1; ++j) {
                fscanf(fp_pic, "%f", &cl_1->inputData[c][i][j]);
            }
        }
    }
    fclose(fp_pic);

    FILE *fp_conv1_w = fopen("/home/xi/params_nin/conv1_w.txt", "r");
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

    FILE *fp_conv1_b = fopen("/home/xi/params_nin/conv1_b.txt", "r");
    for (i = 0; i < cl_1->numOutput; ++i) {
        fscanf(fp_conv1_b, "%f", &cl_1->biasData[i]);
    }
    fclose(fp_conv1_b);

    conv3D(cl_1);

    free(cl_1->inputData);
    free(cl_1->filters);
    free(cl_1->biasData);

    convLayer *cccp_1 = (convLayer *) malloc(sizeof(convLayer));
    int cccp_1_outsize = cl_init(cccp_1, cl_1_outsize, 1, cl_1->numOutput, 96, 0, 1);

    cccp_1->inputData = cl_1->featureMap;
    cccp_1->filters = malloc_4D(cccp_1->numOutput, cccp_1->inChannels, cccp_1->kernelSize, cccp_1->kernelSize);
    cccp_1->biasData = (float *) malloc(sizeof(float) * cccp_1->numOutput); 
    cccp_1->featureMap = malloc_3D(cccp_1->numOutput, cccp_1_outsize, cccp_1_outsize);

    FILE *fp_cccp1_w = fopen("/home/xi/params_nin/cccp1_w.txt", "r");
    for (l = 0; l < cccp_1->numOutput; ++l) {
        for (c = 0; c < cccp_1->inChannels; ++c) {
            for (i = 0; i < cccp_1->kernelSize; ++i) {
                for (j = 0; j < cccp_1->kernelSize; ++j) {
                    fscanf(fp_cccp1_w, "%f", &cccp_1->filters[l][c][i][j]);
                }
            }
        }
    }
    fclose(fp_cccp1_w);

    FILE *fp_cccp1_b = fopen("/home/xi/params_nin/cccp1_b.txt", "r");
    for (i = 0; i < cccp_1->numOutput; ++i) {
        fscanf(fp_cccp1_b, "%f", &cccp_1->biasData[i]);
    }
    fclose(fp_cccp1_b);

    conv3D(cccp_1);

    free(cccp_1->inputData);
    free(cccp_1->filters);
    free(cccp_1->biasData);

    convLayer *cccp_2 = (convLayer *) malloc(sizeof(convLayer));
    int cccp_2_outsize = cl_init(cccp_2, cccp_1_outsize, 1, cccp_1->numOutput, 96, 0, 1);

    cccp_2->inputData = cccp_1->featureMap;
    cccp_2->filters = malloc_4D(cccp_2->numOutput, cccp_2->inChannels, cccp_2->kernelSize, cccp_2->kernelSize);
    cccp_2->biasData = (float *) malloc(sizeof(float) * cccp_2->numOutput); 
    cccp_2->featureMap = malloc_3D(cccp_2->numOutput, cccp_2_outsize, cccp_2_outsize);

    FILE *fp_cccp2_w = fopen("/home/xi/params_nin/cccp2_w.txt", "r");
    for (l = 0; l < cccp_2->numOutput; ++l) {
        for (c = 0; c < cccp_2->inChannels; ++c) {
            for (i = 0; i < cccp_2->kernelSize; ++i) {
                for (j = 0; j < cccp_2->kernelSize; ++j) {
                    fscanf(fp_cccp2_w, "%f", &cccp_2->filters[l][c][i][j]);
                }
            }
        }
    }
    fclose(fp_cccp2_w);

    FILE *fp_cccp2_b = fopen("/home/xi/params_nin/cccp2_b.txt", "r");
    for (i = 0; i < cccp_2->numOutput; ++i) {
        fscanf(fp_cccp2_b, "%f", &cccp_2->biasData[i]);
    }
    fclose(fp_cccp2_b);

    conv3D(cccp_2);

    free(cccp_2->inputData);
    free(cccp_2->filters);
    free(cccp_2->biasData);

    poolLayer *pl_0 = (poolLayer *) malloc(sizeof(poolLayer));

    int pl_0_outsize = pl_init(pl_0, cccp_2->featureMap, cccp_2_outsize, 3, cccp_2->numOutput, 2);
    pl_0->pooledData = malloc_3D(pl_0->inChannels, pl_0_outsize, pl_0_outsize);

    maxPooling(pl_0);

    free(pl_0->inputData);

    convLayer *cl_2 = (convLayer *) malloc(sizeof(convLayer));
    int cl_2_outsize = cl_init(cl_2, pl_0_outsize, 5, pl_0->inChannels, 256, 2, 1);

    cl_2->inputData = malloc_3D(cl_2->inChannels, cl_2->h1 + 2 * cl_2->padding, cl_2->w1 + 2 * cl_2->padding);
    cl_2->filters = malloc_4D(cl_2->numOutput, cl_2->inChannels, cl_2->kernelSize, cl_2->kernelSize);
    cl_2->biasData = (float *) malloc(sizeof(float) * cl_2->numOutput); 
    cl_2->featureMap = malloc_3D(cl_2->numOutput, cl_2_outsize, cl_2_outsize);

    for (c = 0; c < cl_2->inChannels; ++c) {
        for (i = 0; i < cl_2->h1 + 2 * cl_2->padding; ++i) {
            for (j = 0; j < cl_2->w1 + 2 * cl_2->padding; ++j) {
                cl_2->inputData[c][i][j] = 0;
            }
        }
    }
    for (c = 0; c < cl_2->inChannels; ++c) {
        for (i = cl_2->padding; i < cl_2->padding + pl_0_outsize; ++i) {
            for (j = cl_2->padding; j < cl_2->padding + pl_0_outsize; ++j) {
                cl_2->inputData[c][i][j] = pl_0->pooledData[c][i-cl_2->padding][j-cl_2->padding];
            }
        }
    }

    FILE *fp_conv2_w = fopen("/home/xi/params_nin/conv2_w.txt", "r");
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

    FILE *fp_conv2_b = fopen("/home/xi/params_nin/conv2_b.txt", "r");
    for (i = 0; i < cl_2->numOutput; ++i) {
        fscanf(fp_conv2_b, "%f", &cl_2->biasData[i]);
    }
    fclose(fp_conv2_b);

    conv3D(cl_2);

    free(cl_2->inputData);
    free(cl_2->filters);
    free(cl_2->biasData);

    convLayer *cccp_3 = (convLayer *) malloc(sizeof(convLayer));
    int cccp_3_outsize = cl_init(cccp_3, cl_2_outsize, 1, cl_2->numOutput, 256, 0, 1);

    cccp_3->inputData = cl_2->featureMap;
    cccp_3->filters = malloc_4D(cccp_3->numOutput, cccp_3->inChannels, cccp_3->kernelSize, cccp_3->kernelSize);
    cccp_3->biasData = (float *) malloc(sizeof(float) * cccp_3->numOutput); 
    cccp_3->featureMap = malloc_3D(cccp_3->numOutput, cccp_3_outsize, cccp_3_outsize);

    FILE *fp_cccp3_w = fopen("/home/xi/params_nin/cccp3_w.txt", "r");
    for (l = 0; l < cccp_3->numOutput; ++l) {
        for (c = 0; c < cccp_3->inChannels; ++c) {
            for (i = 0; i < cccp_3->kernelSize; ++i) {
                for (j = 0; j < cccp_3->kernelSize; ++j) {
                    fscanf(fp_cccp3_w, "%f", &cccp_3->filters[l][c][i][j]);
                }
            }
        }
    }
    fclose(fp_cccp3_w);

    FILE *fp_cccp3_b = fopen("/home/xi/params_nin/cccp3_b.txt", "r");
    for (i = 0; i < cccp_3->numOutput; ++i) {
        fscanf(fp_cccp3_b, "%f", &cccp_3->biasData[i]);
    }
    fclose(fp_cccp3_b);

    conv3D(cccp_3);

    free(cccp_3->inputData);
    free(cccp_3->filters);
    free(cccp_3->biasData);

    convLayer *cccp_4 = (convLayer *) malloc(sizeof(convLayer));
    int cccp_4_outsize = cl_init(cccp_4, cccp_3_outsize, 1, cccp_3->numOutput, 256, 0, 1);

    cccp_4->inputData = cccp_3->featureMap;
    cccp_4->filters = malloc_4D(cccp_4->numOutput, cccp_4->inChannels, cccp_4->kernelSize, cccp_4->kernelSize);
    cccp_4->biasData = (float *) malloc(sizeof(float) * cccp_4->numOutput); 
    cccp_4->featureMap = malloc_3D(cccp_4->numOutput, cccp_4_outsize, cccp_4_outsize);

    FILE *fp_cccp4_w = fopen("/home/xi/params_nin/cccp4_w.txt", "r");
    for (l = 0; l < cccp_4->numOutput; ++l) {
        for (c = 0; c < cccp_4->inChannels; ++c) {
            for (i = 0; i < cccp_4->kernelSize; ++i) {
                for (j = 0; j < cccp_4->kernelSize; ++j) {
                    fscanf(fp_cccp4_w, "%f", &cccp_4->filters[l][c][i][j]);
                }
            }
        }
    }
    fclose(fp_cccp4_w);

    FILE *fp_cccp4_b = fopen("/home/xi/params_nin/cccp4_b.txt", "r");
    for (i = 0; i < cccp_3->numOutput; ++i) {
        fscanf(fp_cccp4_b, "%f", &cccp_4->biasData[i]);
    }
    fclose(fp_cccp4_b);

    conv3D(cccp_4);

    free(cccp_4->inputData);
    free(cccp_4->filters);
    free(cccp_4->biasData);

    /*for (c = 0; c < cccp_4->numOutput; ++c) {
        for (i = 0; i < cccp_4_outsize; ++i) {
            for (j = 0; j < cccp_4_outsize; ++j) {
                printf("%f ", cccp_4->featureMap[c][i][j]);
            }
        }
    }*/

    poolLayer *pl_2 = (poolLayer *) malloc(sizeof(poolLayer));

    int pl_2_outsize = pl_init(pl_2, cccp_4->featureMap, cccp_4_outsize, 3, cccp_4->numOutput, 2);
    pl_2->pooledData = malloc_3D(pl_2->inChannels, pl_2_outsize, pl_2_outsize);

    maxPooling(pl_2);

    free(pl_2->inputData);

    convLayer *cl_3 = (convLayer *) malloc(sizeof(convLayer));
    int cl_3_outsize = cl_init(cl_3, pl_2_outsize, 3, pl_2->inChannels, 384, 1, 1);

    cl_3->inputData = malloc_3D(cl_3->inChannels, cl_3->h1 + 2 * cl_3->padding, cl_3->w1 + 2 * cl_3->padding);
    cl_3->filters = malloc_4D(cl_3->numOutput, cl_3->inChannels, cl_3->kernelSize, cl_3->kernelSize);
    cl_3->biasData = (float *) malloc(sizeof(float) * cl_3->numOutput); 
    cl_3->featureMap = malloc_3D(cl_3->numOutput, cl_3_outsize, cl_3_outsize);

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

    FILE *fp_conv3_w = fopen("/home/xi/params_nin/conv3_w.txt", "r");
    for (l = 0; l < cl_3->numOutput; ++l) {
        for (c = 0; c < cl_3->inChannels; ++c) {
            for (i = 0; i < cl_3->kernelSize; ++i) {
                for (j = 0; j < cl_3->kernelSize; ++j) {
                    fscanf(fp_conv3_w, "%f", &cl_3->filters[l][c][i][j]);
                }
            }
        }
    }
    fclose(fp_conv3_w);

    FILE *fp_conv3_b = fopen("/home/xi/params_nin/conv3_b.txt", "r");
    for (i = 0; i < cl_3->numOutput; ++i) {
        fscanf(fp_conv3_b, "%f", &cl_3->biasData[i]);
    }
    fclose(fp_conv3_b);

    conv3D(cl_3);

    free(cl_3->inputData);
    free(cl_3->filters);
    free(cl_3->biasData);

    convLayer *cccp_5 = (convLayer *) malloc(sizeof(convLayer));
    int cccp_5_outsize = cl_init(cccp_5, cl_3_outsize, 1, cl_3->numOutput, 384, 0, 1);

    cccp_5->inputData = cl_3->featureMap;
    cccp_5->filters = malloc_4D(cccp_5->numOutput, cccp_5->inChannels, cccp_5->kernelSize, cccp_5->kernelSize);
    cccp_5->biasData = (float *) malloc(sizeof(float) * cccp_5->numOutput); 
    cccp_5->featureMap = malloc_3D(cccp_5->numOutput, cccp_5_outsize, cccp_5_outsize);

    FILE *fp_cccp5_w = fopen("/home/xi/params_nin/cccp5_w.txt", "r");
    for (l = 0; l < cccp_5->numOutput; ++l) {
        for (c = 0; c < cccp_5->inChannels; ++c) {
            for (i = 0; i < cccp_5->kernelSize; ++i) {
                for (j = 0; j < cccp_5->kernelSize; ++j) {
                    fscanf(fp_cccp5_w, "%f", &cccp_5->filters[l][c][i][j]);
                }
            }
        }
    }
    fclose(fp_cccp5_w);

    FILE *fp_cccp5_b = fopen("/home/xi/params_nin/cccp5_b.txt", "r");
    for (i = 0; i < cccp_5->numOutput; ++i) {
        fscanf(fp_cccp5_b, "%f", &cccp_5->biasData[i]);
    }
    fclose(fp_cccp5_b);

    conv3D(cccp_5);

    free(cccp_5->inputData);
    free(cccp_5->filters);
    free(cccp_5->biasData);

    convLayer *cccp_6 = (convLayer *) malloc(sizeof(convLayer));
    int cccp_6_outsize = cl_init(cccp_6, cccp_5_outsize, 1, cccp_5->numOutput, 384, 0, 1);

    cccp_6->inputData = cccp_5->featureMap;
    cccp_6->filters = malloc_4D(cccp_6->numOutput, cccp_6->inChannels, cccp_6->kernelSize, cccp_6->kernelSize);
    cccp_6->biasData = (float *) malloc(sizeof(float) * cccp_6->numOutput); 
    cccp_6->featureMap = malloc_3D(cccp_6->numOutput, cccp_6_outsize, cccp_6_outsize);

    FILE *fp_cccp6_w = fopen("/home/xi/params_nin/cccp6_w.txt", "r");
    for (l = 0; l < cccp_6->numOutput; ++l) {
        for (c = 0; c < cccp_6->inChannels; ++c) {
            for (i = 0; i < cccp_6->kernelSize; ++i) {
                for (j = 0; j < cccp_6->kernelSize; ++j) {
                    fscanf(fp_cccp6_w, "%f", &cccp_6->filters[l][c][i][j]);
                }
            }
        }
    }
    fclose(fp_cccp6_w);

    FILE *fp_cccp6_b = fopen("/home/xi/params_nin/cccp6_b.txt", "r");
    for (i = 0; i < cccp_6->numOutput; ++i) {
        fscanf(fp_cccp6_b, "%f", &cccp_6->biasData[i]);
    }
    fclose(fp_cccp6_b);

    conv3D(cccp_6);

    free(cccp_6->inputData);
    free(cccp_6->filters);
    free(cccp_6->biasData);
    
    poolLayer *pl_3 = (poolLayer *) malloc(sizeof(poolLayer));

    int pl_3_outsize = pl_init(pl_3, cccp_6->featureMap, cccp_6_outsize, 3, cccp_6->numOutput, 2);
    pl_3->pooledData = malloc_3D(pl_3->inChannels, pl_3_outsize, pl_3_outsize);

    maxPooling(pl_3);

    free(pl_3->inputData);

    convLayer *cl_4 = (convLayer *) malloc(sizeof(convLayer));
    int cl_4_outsize = cl_init(cl_4, pl_3_outsize, 3, pl_3->inChannels, 1024, 1, 1);

    cl_4->inputData = malloc_3D(cl_4->inChannels, cl_4->h1 + 2 * cl_4->padding, cl_4->w1 + 2 * cl_4->padding);
    cl_4->filters = malloc_4D(cl_4->numOutput, cl_4->inChannels, cl_4->kernelSize, cl_4->kernelSize);
    cl_4->biasData = (float *) malloc(sizeof(float) * cl_4->numOutput); 
    cl_4->featureMap = malloc_3D(cl_4->numOutput, cl_4_outsize, cl_4_outsize);

    for (c = 0; c < cl_4->inChannels; ++c) {
        for (i = 0; i < cl_4->h1 + 2 * cl_4->padding; ++i) {
            for (j = 0; j < cl_4->w1 + 2 * cl_4->padding; ++j) {
                cl_4->inputData[c][i][j] = 0;
            }
        }
    }
    for (c = 0; c < cl_4->inChannels; ++c) {
        for (i = cl_4->padding; i < cl_4->padding + pl_3_outsize; ++i) {
            for (j = cl_4->padding; j < cl_4->padding + pl_3_outsize; ++j) {
                cl_4->inputData[c][i][j] = pl_3->pooledData[c][i-cl_4->padding][j-cl_4->padding];
            }
        }
    }

    FILE *fp_conv4_w = fopen("/home/xi/params_nin/conv4_w.txt", "r");
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

    FILE *fp_conv4_b = fopen("/home/xi/params_nin/conv4_b.txt", "r");
    for (i = 0; i < cl_4->numOutput; ++i) {
        fscanf(fp_conv4_b, "%f", &cl_4->biasData[i]);
    }
    fclose(fp_conv4_b);

    conv3D(cl_4);

    free(cl_4->inputData);
    free(cl_4->filters);
    free(cl_4->biasData);

    convLayer *cccp_7 = (convLayer *) malloc(sizeof(convLayer));
    int cccp_7_outsize = cl_init(cccp_7, cl_4_outsize, 1, cl_4->numOutput, 1024, 0, 1);

    cccp_7->inputData = cl_4->featureMap;
    cccp_7->filters = malloc_4D(cccp_7->numOutput, cccp_7->inChannels, cccp_7->kernelSize, cccp_7->kernelSize);
    cccp_7->biasData = (float *) malloc(sizeof(float) * cccp_7->numOutput); 
    cccp_7->featureMap = malloc_3D(cccp_7->numOutput, cccp_7_outsize, cccp_7_outsize);

    FILE *fp_cccp7_w = fopen("/home/xi/params_nin/cccp7_w.txt", "r");
    for (l = 0; l < cccp_7->numOutput; ++l) {
        for (c = 0; c < cccp_7->inChannels; ++c) {
            for (i = 0; i < cccp_7->kernelSize; ++i) {
                for (j = 0; j < cccp_7->kernelSize; ++j) {
                    fscanf(fp_cccp7_w, "%f", &cccp_7->filters[l][c][i][j]);
                }
            }
        }
    }
    fclose(fp_cccp7_w);

    FILE *fp_cccp7_b = fopen("/home/xi/params_nin/cccp7_b.txt", "r");
    for (i = 0; i < cccp_7->numOutput; ++i) {
        fscanf(fp_cccp7_b, "%f", &cccp_7->biasData[i]);
    }
    fclose(fp_cccp7_b);

    conv3D(cccp_7);

    free(cccp_7->inputData);
    free(cccp_7->filters);
    free(cccp_7->biasData);

    convLayer *cccp_8 = (convLayer *) malloc(sizeof(convLayer));
    int cccp_8_outsize = cl_init(cccp_8, cccp_7_outsize, 1, cccp_7->numOutput, 50, 0, 1);
    cccp_8->inputData = cccp_7->featureMap;
    cccp_8->filters = malloc_4D(cccp_8->numOutput, cccp_8->inChannels, cccp_8->kernelSize, cccp_8->kernelSize);
    cccp_8->biasData = (float *) malloc(sizeof(float) * cccp_8->numOutput); 
    cccp_8->featureMap = malloc_3D(cccp_8->numOutput, cccp_8_outsize, cccp_8_outsize);

    FILE *fp_cccp8_w = fopen("/home/xi/params_nin/cccp8_w.txt", "r");
    for (l = 0; l < cccp_8->numOutput; ++l) {
        for (c = 0; c < cccp_8->inChannels; ++c) {
            for (i = 0; i < cccp_8->kernelSize; ++i) {
                for (j = 0; j < cccp_8->kernelSize; ++j) {
                    fscanf(fp_cccp8_w, "%f", &cccp_8->filters[l][c][i][j]);
                }
            }
        }
    }
    fclose(fp_cccp8_w);

    FILE *fp_cccp8_b = fopen("/home/xi/params_nin/cccp8_b.txt", "r");
    for (i = 0; i < cccp_8->numOutput; ++i) {
        fscanf(fp_cccp8_b, "%f", &cccp_8->biasData[i]);
    }
    fclose(fp_cccp8_b);

    conv3D(cccp_8);

    free(cccp_8->inputData);
    free(cccp_8->filters);
    free(cccp_8->biasData);

    poolLayer *pl_4 = (poolLayer *) malloc(sizeof(poolLayer));
    int pl_4_outsize = pl_init(pl_4, cccp_8->featureMap, cccp_8_outsize, cccp_8_outsize, cccp_8->numOutput, 1);
    pl_4->inputData = cccp_8->featureMap;
    pl_4->pooledData = malloc_3D(pl_4->inChannels, pl_4_outsize, pl_4_outsize);

    averagePooling(pl_4);

    free(pl_4->inputData);

    for (c = 0; c < pl_4->inChannels; ++c) {
        for (i = 0; i < pl_4_outsize; ++i) {
            for (j = 0; j < pl_4_outsize; ++j) {
                printf("%f ", pl_4->pooledData[c][i][j]);
            }
        }
    }
    return 0;
}
