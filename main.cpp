#include <cmath>
#include <stdio.h>
#include <cstdlib>
#include <time.h>
using namespace std;

//constants
const double dimension[2] = {28*28, 10};
const double w_limit = sqrt(6/(dimension[0]+dimension[1]));
//[layers][b or w][lower_limit or upper_limit]
const double distribution[2][2][2] = {
    {{0,0},{0,0}},
    {{0,0},{-w_limit,w_limit}}
};

//use for initialize dynamic array
class dynamic_arr{
public:
    double* oneD_init(size_t size, double *arr){
        arr = (double *)calloc(size, sizeof(double));
        if(arr==NULL){
            printf("Array not allocated!");
            exit(1);
        }
        return arr;
    }
    /*double* oneD_resize(size_t size, double* arr){
        arr = (double *)realloc(arr, size * sizeof(double));
        if(arr==NULL){
            printf("Array not allocated!");
            exit(1);
        }
        return arr;
    }*/
    double** twoD_init(size_t m, size_t n, double **arr){
        arr = (double**)calloc(m, sizeof(double*));
        for(int i=0; i<m; i++){
            arr[i] = (double*)calloc(n, sizeof(double));
        }
        if(arr==NULL){
            printf("Array not allocated!");
            exit(1);
        }
        return arr;
    }
    double*** threeD_init(size_t m, size_t n, size_t o, double ***arr){
        arr = (double***)calloc(m, sizeof(double**));
        for(int i=0; i<m; i++){
            arr[i] = (double**)calloc(n, sizeof(double*));
            for(int j=0; j<n; j++){
                arr[i][j] = (double*)calloc(o, sizeof(double));
            }
        }
        if(arr==NULL){
            printf("Array not allocated!");
            exit(1);
        }
        return arr;
    }
}d_arr;

double* softmax(size_t size, double* x){
    // softmax: output[i] = exp(i)/summation all exp(var)
    double* x_exp,* result;
    double sum;
    x_exp = d_arr.oneD_init(size, x_exp);
    result = d_arr.oneD_init(size, result);
    for(int i=0;i<size;i++){
        x_exp[i] = exp(x[i]);
        sum = sum + x_exp[i];
    }
    for(int i=0;i<size;i++){
        result[i] = x_exp[i]/sum;
    }
    free(x_exp);
    return result;
}

double* init_parameters_b(int layer){
    //random seed
    srand(time(0));

    double dist[2];
    double* rand_b;
    rand_b = d_arr.oneD_init(dimension[layer],rand_b);
    dist[0] = distribution[layer][0][0];
    dist[1] = distribution[layer][0][1];

    for(int i=0;i<dimension[layer];i++){
        rand_b[i] = (dist[1] - dist[0]) * rand() / (RAND_MAX + 1.0) + dist[0];
    }
    return rand_b;
}

double** init_parameters_w(int layer){
    //random seed
    srand(time(0));

    double dist[2];
    double** rand_w;
    rand_w = d_arr.twoD_init(dimension[layer-1], dimension[layer], rand_w);
    dist[0] = distribution[layer][1][0];
    dist[1] = distribution[layer][1][1];

    for(int i=0;i<dimension[layer-1];i++){
        for(int j=0; j<dimension[layer]; j++){
            rand_w[i][j] = (dist[1] - dist[0]) * rand() / (RAND_MAX + 1.0) + dist[0];
        }
    }
    return rand_w;
}

double **twoD_multi(int m, int n, int o, int p, double *A, double **B){
    //m*n times o*p = m*p
    //1x784 * 784*10 = 1*10
    if(n!=o){
        printf("\nMatrix multiplication error!");
        exit(1);
    }

    double **product;
    product = d_arr.twoD_init(m, p, product);

    //row x column
    for(int i=0; i<m; i++){
        for(int j=0; j<p;j++){
            for(int k=0; k<n; k++){
                product[i][j] = product[i][j] +  A[k] * B[k][j];
            }
        }
    }
    return product;
} 

double *predict(double *img, double **b, double ***w){
    //l0 = tanh(data + b0)
    //l1 = softmax(l0 x W + b1)
    double *l0, *l1;
    double **product;
    l0 = d_arr.oneD_init(dimension[0],l0);
    l1 = d_arr.oneD_init(dimension[1],l1);

    //step 1
    for(int i=0;i<dimension[0];i++){
        l0[i] = tanh(img[i] + b[0][i]);
    }

    //step 2
    //pass the l0 and the second layer of weight to product function
    product = twoD_multi(1, dimension[0], dimension[0], dimension[1], l0, w[1]);
    for(int i=0;i<dimension[1];i++){
        l1[i] = product[0][i];
    }
    l1 = softmax(dimension[1], l1);
    return l1;
}

int main(){
    //initialization of parameters
    double **b, ***w;
    int *address[2];
    b = d_arr.twoD_init(2,dimension[0], b);
    w = d_arr.threeD_init(2, dimension[0], dimension[1], w);
    for(int i=0; i<2;i++){
        b[i] = init_parameters_b(i);
        if(i!=0){
            w[i] = init_parameters_w(i);
        }    
    }

    //gernerate random noise
    double *noise;
    noise = d_arr.oneD_init(dimension[0], noise);
    srand(time(0));
    for(int i=0; i<dimension[0];i++){
        noise[i] = (1 - 0) * rand() / (RAND_MAX + 1.0) + 0;
    }
    
    //get prediction
    double *result;
    result = d_arr.oneD_init(dimension[1], result);
    result = predict(noise, b, w);

    //print all variable and find the max
    double max = result[0];
    int n = 0;
    for(int i=0;i<dimension[1];i++){
        if(result[i]>max){
            max = result[i];
            n = i;
        }
        printf("%f, ", result[i]);
    }
    printf("\nPrediction: %d", n+1);
    
    //release ram
    free(noise);
    free(result);
    free(b);
    free(w);
    return 0;
}
