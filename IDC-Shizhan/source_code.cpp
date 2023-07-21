#include <stdio.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <immintrin.h>
#include <cstdlib>
using namespace std;

typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;
auto max_threads=std::thread::hardware_concurrency();
int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0, F2 = 0;

float *X0, *W1, *W2, *X1, *X1_inter, *X2, *X2_inter;
std::vector<int> rowPtr;
std::vector<int> colIndex;
std::vector<float> edge_value;
vector<int> raw_graph;

void edgeNormalization()
{   
#pragma omp parallel for num_threads(max_threads-1)
    for (int i = 0; i < v_num; i++)
    {
        int start = rowPtr[i];
        int end = rowPtr[i + 1];
        int degree = end - start; // 计算度信息
        
        for (int j = start; j < end; j++)
        {
            int src = i;
            int dst = colIndex[j];
            float val = 1.0 / sqrt(degree) / sqrt(rowPtr[dst + 1] - rowPtr[dst]);
            edge_value[j] = val;
        }
    }
}

void readGraph(char *fname)
{
	ifstream infile(fname);

	int source;
	int end;

	infile >> v_num >> e_num;

	// raw_graph.resize(e_num * 2);

	while (!infile.eof())
	{
		infile >> source >> end;
		if (infile.peek() == EOF)
			break;
		raw_graph.push_back(source);
		raw_graph.push_back(end);  // 基数sourse 偶数end 长度为2*enum
	}
}


void convertToCSR() {
    rowPtr.resize(v_num + 1, 0);
    colIndex.resize(e_num);
    edge_value.resize(e_num);

    #pragma omp parallel for num_threads(max_threads-1)
    for (int i = 0; i < raw_graph.size() / 2; i++) {
        auto t_id=std::this_thread::get_id();
        
        int src = raw_graph[2 * i];
        int dst = raw_graph[2 * i + 1];
        
    #pragma omp atomic
        rowPtr[src + 1]++;
        colIndex[i] = dst;
        // Assign edge value if needed: edge_value[i] = ...;
    }

    for (int i = 1; i <= v_num; i++) {
        rowPtr[i] += rowPtr[i - 1];
    }
}


void readFloat(char *fname, float *&dst, int num)
{
    dst = (float *)malloc(num * sizeof(float));    
    FILE *fp = fopen(fname, "rb");
    fread(dst, num * sizeof(float), 1, fp);
    fclose(fp);
}


void initFloat(float *&dst, int num)
{
	dst = (float *)malloc(num * sizeof(float));
	memset(dst, 0, num * sizeof(float));  // 初始化为0
}

void XW(int in_dim, int out_dim, float *in_X, float *out_X, float *W)
{
	float(*tmp_in_X)[in_dim] = (float(*)[in_dim])in_X;
	float(*tmp_out_X)[out_dim] = (float(*)[out_dim])out_X;
	float(*tmp_W)[out_dim] = (float(*)[out_dim])W;


	const int SIMD_NUM=8;
	int n = out_dim/SIMD_NUM;//循环次数
	#pragma omp parallel for num_threads(max_threads-1) //schedule(dynamic)
	for (int i = 0; i < v_num; i++)
	{
		for (int k = 0; k < in_dim; k++)
		{
			//__m256 x=_mm256_set1_ps(tmp_in_X[i][k]);
			__m256 x=_mm256_broadcast_ss(reinterpret_cast<float const*>(&(tmp_in_X[i][k])));
			for(int j=0;j<n;j++){
				__m256 mul_in_w=_mm256_mul_ps(x,_mm256_loadu_ps(reinterpret_cast<float const*>(&(tmp_W[k][j*SIMD_NUM]))));
				__m256 old_out=_mm256_loadu_ps(reinterpret_cast<float const*>(&(tmp_out_X[i][j*SIMD_NUM])));
				_mm256_storeu_ps(&(tmp_out_X[i][j*SIMD_NUM]),_mm256_add_ps(old_out,mul_in_w));
			}
			//剩下的零散部分处理
			for (int j = n*SIMD_NUM; j < out_dim; j++)
			{
				tmp_out_X[i][j] += tmp_in_X[i][k] * tmp_W[k][j];
			}
		}
	}
}

void AX(int dim, float* in_X, float* out_X)
{
    float(*tmp_in_X)[dim] = (float(*)[dim])in_X;
    float(*tmp_out_X)[dim] = (float(*)[dim])out_X;

	const int SIMD_NUM=8;
	int n=dim/SIMD_NUM;
	#pragma omp parallel for num_threads(max_threads-1) // schedule(dynamic)
    for (int i = 0; i < v_num; i++)
    {
        int start = rowPtr[i];
        int end = rowPtr[i + 1];
        for (int j = start; j < end; j++) //nbr是i的邻居节点
        {
            int nbr = colIndex[j];
			//__m256 w=_mm256_set1_ps(edge_value[j]);
            __m256 w=_mm256_broadcast_ss(reinterpret_cast<float const*>(&(edge_value[j])));
			for (int k=0;k<n;k++){
				__m256 in=_mm256_loadu_ps(reinterpret_cast<float const*>(&(tmp_in_X[nbr][k*SIMD_NUM])));
				__m256 out=_mm256_loadu_ps(reinterpret_cast<float const*>(&(tmp_out_X[i][k*SIMD_NUM])));
				_mm256_storeu_ps(&(tmp_out_X[i][k*SIMD_NUM]),_mm256_add_ps(_mm256_mul_ps(in,w),out));
			}
            for (int k = SIMD_NUM*n; k < dim; k++)
            {
                tmp_out_X[i][k] += tmp_in_X[nbr][k] * edge_value[j];
            }
        }
    }
}

void ReLU(int dim, float *X)
{
#pragma omp parallel for num_threads(max_threads-1)
	for (int i = 0; i < v_num * dim; i++)
		if (X[i] < 0)
			X[i] = 0;
}


void LogSoftmax(int dim, float *X)
{
	float(*tmp_X)[dim] = (float(*)[dim])X;
#pragma omp parallel for num_threads(max_threads-1)
	for (int i = 0; i < v_num; i++)
	{
		float max = tmp_X[i][0];
		for (int j = 1; j < dim; j++)
		{
			if (tmp_X[i][j] > max)
				max = tmp_X[i][j];
		}

		float sum = 0;
		for (int j = 0; j < dim; j++)
		{
			sum += exp(tmp_X[i][j] - max);
		}
		sum = log(sum);

		for (int j = 0; j < dim; j++)
		{
			tmp_X[i][j] = tmp_X[i][j] - max - sum;
		}
	}
}




float MaxRowSum(float *X, int dim)
{
	float(*tmp_X)[dim] = (float(*)[dim])X;
	float max = -__FLT_MAX__;
    //#pragma omp parallel for num_threads(max_threads-1)
	for (int i = 0; i < v_num; i++)
	{
		float sum = 0;
		for (int j = 0; j < dim; j++)
		{
			sum += tmp_X[i][j];
		}
		if (sum > max)
			max = sum;
	}
	return max;
}

void freeFloats()
{
	free(X0);
	free(W1);
	free(W2);
	free(X1);
	free(X2);
	free(X1_inter);
	free(X2_inter);
}

void somePreprocessing()
{
	//The graph  will be transformed into adjacency list ,you can use other data structure such as CSR
	convertToCSR();
}

int main(int argc, char **argv)
{
	// Do NOT count the time of reading files, malloc, and memset
	F0 = atoi(argv[1]); // 64
	F1 = atoi(argv[2]); // 16
	F2 = atoi(argv[3]); // 8

	readGraph(argv[4]); // graph/1024_example_graph.txt
	readFloat(argv[5], X0, v_num * F0);  // embedding/1024.bin 
	readFloat(argv[6], W1, F0 * F1);  // weight/W_64_16.bin
	readFloat(argv[7], W2, F1 * F2);  // weight/W_16_8.bin

	initFloat(X1, v_num * F1);
	initFloat(X1_inter, v_num * F1);
	initFloat(X2, v_num * F2);
	initFloat(X2_inter, v_num * F2);

	// Time point at the start of the computation
	TimePoint start = chrono::steady_clock::now();

	// Preprocessing time should be included

	somePreprocessing();

	edgeNormalization();
	
	// printf("Layer1 XW\n");
	XW(F0, F1, X0, X1_inter, W1);

	// printf("Layer1 AX\n");
	AX(F1, X1_inter, X1);

	ReLU(F1, X1);

	// printf("Layer2 XW\n");
	XW(F1, F2, X1, X2_inter, W2);

	// printf("Layer2 AX\n");
	AX(F2, X2_inter, X2);

	// printf("Layer2 LogSoftmax\n");
	LogSoftmax(F2, X2);

	// You need to compute the max row sum for result verification
	float max_sum = MaxRowSum(X2, F2);

	// Time point at the end of the computation
	TimePoint end = chrono::steady_clock::now();
	chrono::duration<double> l_durationSec = end - start;
	double l_timeMs = l_durationSec.count() * 1e3;  // ms

	// Finally, the max row sum and the computing time
	// should be print to the terminal in the following format
	printf("%.8f\n", max_sum);
	printf("%.8lf\n", l_timeMs);

	// Remember to free your allocated memory
	freeFloats();
}
