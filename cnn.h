typedef struct convolutional_layer {
    int w1;   //输入图像的宽
    int h1;  //输入图像的长
    int stride; //步长
    int kernelSize;      //卷积核的大小，一般是正方形
    int inChannels;      //输入通道数
    int numOutput;       //输出通道数
    int padding;      //零填充的大小
    
    //输入图片数据:通道数*宽度*高度
    float ***inputData;

    //filter个数*channels*kernelSize*kernelSize
    float ****filters;

    // FeatureMap的数据，这里是一个三维数组
    // 其大小为filter*mapSize*mapSize大小
    float ***featureMap;   
    float *biasData;   //偏置，偏置的大小为filter的个数
} convLayer;

typedef struct pooling_layer {
    int w1;   //输入图像的宽
    int h1;  //输入图像的长
    int kernelSize;      //池化核的大小
    int stride;       //步长
    int inChannels;   //输入图像的数量

    float ***inputData;
    float ***pooledData;
} poolLayer;


// 输出层 全连接的神经网络
typedef struct nn_layer {
    int inputNum;   //输入数据的数目
    int outputNum;  //输出数据的数目

    float **weightData; // 权重数据，为一个inputNum*outputNum大小
    float *biasData;   //偏置，大小为outputNum大小

    float *inputData;
    float *outputData;
    // 下面三者的大小同输出的维度相同
    float *v; // 进入激活函数的输入值
    float *y; // 激活函数后神经元的输出
    float *d; // 网络的局部梯度,δ值
} fcLayer;

//完整卷积神经网络
typedef struct cnn_network {
    int layerNum;
    convLayer *C1;
    poolLayer *S2;
    convLayer *C3;
    poolLayer *S4;
    fcLayer *O5;

    float *e; // 训练误差
    float *L; // 瞬时误差能量
} cnn;

//暂时存放参数
typedef struct train_opts {
    int num_epochs; // 训练的迭代次数
    float lr; // 学习速率
} cnnOpts;

float relu(float x);
void conv3D(convLayer *conv); //3D卷积计算函数
void maxPooling(poolLayer *pool); //最大池化
void pooled2fc(poolLayer pool, float ****pooled2fcWeights, fcLayer *fc); //计算从池化层到全连接层
float locateSub(float **a, int m, int n, int stride, int kernelSize); //用于计算最大池化的辅助函数
void fc2fc(fcLayer *fc); //全连接到全连接
float conv_helper(convLayer *conv, int channels, int loc_m, int loc_n, int stride, int nth_filter); //卷积计算的辅助函数