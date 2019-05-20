
#include "tbb/task_scheduler_init.h"
#include"tbb/blocked_range.h"
#include"tbb/parallel_for.h"
#include "tbb/tick_count.h"
#include <iostream>

using namespace tbb;

#define EPSILON 1.0E-20
#define MAX_ITERATIONS 100000

void ÑalculationNEwX(double *x, double *d, double s, int size) {
	int i;

	for (i = 0; i < size; i++)
		x[i] = x[i] + s*d[i];
}
double Vectorsmultiplication(double *Vec1, double *Vec2, int size) {
	int i;
	double res;

	for (i = 0, res = 0; i < size; i++)
		res += Vec1[i] * Vec2[i];

	return res;
}
/*/
void BlocVectormultiplication(double *Bloc, double *Vec, double *VecR, int sizeB, int sizeV) {
	int i, j, sizeR;

	sizeR = sizeB / sizeV;

	for (int i = 0; i < sizeR; i++)
		VecR[i] = 0;

	for (i = 0; i < sizeR; i++)
		for (j = 0; j < sizeV; j++)
			VecR[i] += Bloc[i*sizeV + j] * Vec[j];
}
*/
void CreateTask(int size, double** A, double *_a, double* B, double *g_count, double *tmpr) {
	int i, j, k;

	//**/srand(time(NULL));

	for (i = 0; i < size; i++)
		for (j = 0; j <= i; j++) {
			A[i][j] = (double)(rand() % 10);
			A[j][i] = A[i][j];
		}

	k = 0;
	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++) {
			_a[k] = A[i][j];
			k++;
		}

	for (i = 0; i < size; i++) {
		B[i] = (double)(rand() % 10);
		g_count[i] = 0;
		tmpr[i] = 0;
	}
}


class MatrixVectorMultiplicator{
	double *pMatrix, *pVector, *pResult;
	const int Size;
public:
	void operator() (const blocked_range<int>& r)const {
		int begin = r.begin(); 
		int end = r.end();
		for (int i = begin; i != end; i++) {
			pResult[i] = 0;
			for (int j = 0; j<Size; j++)
				pResult[i] += pMatrix[i*Size + j] * pVector[j];
		}
	}
	MatrixVectorMultiplicator(double *pm, double *pv, double *pr, int sz) :pMatrix(pm), pVector(pv), pResult(pr), Size(sz) {}
};

void ParallelResultCalculation(double* pMatrix, double* pVector, double* pResult, int Size, int grainsize) {
	parallel_for(blocked_range<int>(0, Size, grainsize), MatrixVectorMultiplicator(pMatrix, pVector, pResult, Size));
}

int main() {

	double *pMatrix;	//The first argument-initial matrix
	double *pVector;	//The second argument-initial vector
	double *pResult;	//Result vector for matrix-vector multiplication
	int Size;			//Sizes of initial matrix and vector



	int task_size, grandsize;// , size_block_elem;
	int i, iter;
	tick_count StartTime, EndTime;
	double division;
	double eps;
	double **MatrA, *A, *B, *X;
	double s, *d_prev, *g_prev, *g_count;
	double *tmpres;
	int yyy = 1;
	while (true) {
		std::cout << "Size of system: ";
		std::cin >> task_size; 
		std::cin >> yyy;

		//	size_block_elem = task_size*task_size;

		X = new double[task_size];
		d_prev = new double[task_size];
		MatrA = new double*[task_size];
		for (int i = 0; i < task_size; i++)
			MatrA[i] = new double[task_size];
		A = new double[task_size*task_size];
		B = new double[task_size];
		g_prev = new double[task_size];
		g_count = new double[task_size];
		tmpres = new double[task_size];

		CreateTask(task_size, MatrA, A, B, g_count, tmpres);

		for (i = 0; i < task_size; i++) {
			X[i] = 0.0;
			g_prev[i] = B[i];
			d_prev[i] = g_prev[i];
		}

		StartTime = tick_count::now();

		iter = 0;
		
		grandsize = task_size / yyy;
		std::cout << "grandsize: " << grandsize << std::endl;
		do {

			//-------------------------------------------------------------------------//
			//--------------[ s^k = <g^k-1,g^k-1> / (d^k-1 * A * d^k-1) ]--------------//
			//-------------------------------------------------------------------------//

			//	ParallelResultCalculation(pMatrix, pVector, pResult, Size, Size);

			task_scheduler_init init(1);
			ParallelResultCalculation(A, d_prev, tmpres, task_size, grandsize);
			init.terminate();

			//		BlocVectormultiplication(A, d_prev, tmpres, size_block_elem, task_size);

			s = Vectorsmultiplication(g_prev, g_prev, task_size) / Vectorsmultiplication(tmpres, d_prev, task_size);

			//-------------------------------------------------------------------------//
			//---------------------[ x^k = x^(k-1) + s^k * d^k-1 ]---------------------//
			//-------------------------------------------------------------------------//

			ÑalculationNEwX(X, d_prev, s, task_size);

			//-------------------------------------------------------------------------//
			//----------------------[ g^k = g^k-1  -  s*A*d^k-1 ]----------------------//
			//-------------------------------------------------------------------------//

			for (i = 0; i < task_size; i++)
				g_count[i] = g_prev[i] - s * tmpres[i];

			eps = Vectorsmultiplication(g_count, g_count, task_size);

			//-------------------------------------------------------------------------//
			//----------[ d^k = g^k + <g^k,g^k>/<g^(k-1),g^(k-1)> * d^(k-1) ]----------//
			//-------------------------------------------------------------------------//

			division = Vectorsmultiplication(g_count, g_count, task_size) / Vectorsmultiplication(g_prev, g_prev, task_size);

			for (i = 0; i < task_size; i++) {
				d_prev[i] = g_count[i] + division*d_prev[i];
				g_prev[i] = g_count[i];
			}

			iter++;

		} while (iter < MAX_ITERATIONS && eps > EPSILON);

		EndTime = tick_count::now();

		std::cout << "Num of Iteration: " << iter << ".\nTime: " << ((double)(EndTime - StartTime).seconds()) << std::endl;
		if (task_size <= 5) {
			std::cout << "Matrix:\n";
			for (int i = 0; i < task_size; i++) {
				for (int j = 0; j < task_size; j++)
					std::cout << MatrA[i][j] << " ";
				std::cout << std::endl;
			}
			std::cout << "B:\n";
			for (int i = 0; i < task_size; i++)
				std::cout << B[i] << " ";
			std::cout << "\nX is:\n";
			for (int i = 0; i < task_size; i++)
				std::cout << X[i] << " ";
			std::cout << std::endl << std::endl;
		}
		
	}
	return 0;
}