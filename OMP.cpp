//Решение систем линейных уравнений методом сопряженных градиентов

#include <iostream>
#include <ctime>
#include <omp.h>

#define EPSILON 1.0E-20
#define MAX_ITERATIONS 100000

void СalculationNEwX(double *x, double *d, double s, int size) {
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
void BlocVectormultiplication(double *Bloc, double *Vec, double *VecR, int sizeB, int sizeV) {
	int i, j, sizeR;
//		int check;
	double sum;

		sizeR = sizeB / sizeV;

		for (int i = 0; i < sizeR; i++)
			VecR[i] = 0;

//		#pragma omp paralell shared (i, j, sizeR,Bloc, Vec, VecR, sizeB, sizeV, sum) 
	//	{
//			std::cout << "Kol-vo potokov: " << omp_get_num_threads() << std::endl;
			//	#pragma omp paralell for private (j)
			//	{
			//		std::cout << "Hi, I am " << omp_get_thread_num() << std::endl;
			for (i = 0; i < sizeR; i++) {
				sum = 0.0;
				#pragma omp parallel for reduction(+:sum)
				for (j = 0; j < sizeV; j++) {
					//	std::cout << "Hi, I am " << omp_get_thread_num() << std::endl;
						//VecR[i] += Bloc[i*sizeV + j] * Vec[j];
					sum += Bloc[i*sizeV + j] * Vec[j];
					//if (omp_get_thread_num() == 0){ std::cout << "Kol-vo potokov: " << omp_get_num_threads() << std::endl; }
					//check = omp_get_num_threads();
				}

				VecR[i] = sum;
			}
			//	}
//		}
		
}
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

int main(int argc, char *argv[]) {

//	int i, j;
//	std::cout << "Num of free proc: " << omp_get_num_procs() << std::endl;
//	std::cout << "Set threads: ";
//	std::cin >> i;
//	omp_set_num_threads(i);
	//std::cout << "Num of free threads: " << omp_get_num_threads() << std::endl;
//	std::cout << "Num of free threads: " << omp_get_max_threads() << std::endl;
//	std::cout << "My thread id: " << omp_get_thread_num() << std::endl;
//#pragma omp parallel private(j)
//	{
//	j = omp_get_thread_num();
//	if (j == 0) 
//		std::cout << "Num of free threads: " << omp_get_num_threads() << std::endl;
//	}


	int task_size, size_block_elem;
	int i, iter;
	clock_t StartTime, EndTime;
	double division;
	double eps;

	double **MatrA, *A, *B, *X;
	double s, *d_prev, *g_prev, *g_count;
	double *tmpres;
	while (true) {

		std::cout << "Set threads: ";
		std::cin >> i;
		omp_set_num_threads(i);

		std::cout << "Size of system: ";
		std::cin >> task_size;

		size_block_elem = task_size*task_size;

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

		StartTime = clock();

		iter = 0;

		do {

			//-------------------------------------------------------------------------//
			//--------------[ s^k = <g^k-1,g^k-1> / (d^k-1 * A * d^k-1) ]--------------//
			//-------------------------------------------------------------------------//

			BlocVectormultiplication(A, d_prev, tmpres, size_block_elem, task_size);

			s = Vectorsmultiplication(g_prev, g_prev, task_size) / Vectorsmultiplication(tmpres, d_prev, task_size);

			//-------------------------------------------------------------------------//
			//---------------------[ x^k = x^(k-1) + s^k * d^k-1 ]---------------------//
			//-------------------------------------------------------------------------//

			СalculationNEwX(X, d_prev, s, task_size);

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


		EndTime = clock();

		std::cout << "Num of Iteration: " << iter << ".\nTime: " << ((double)(EndTime - StartTime)) / CLOCKS_PER_SEC << std::endl;
	/*	std::cout << "Matrix:\n";
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
*/
	//	break;
	}
	
	return 0;
}
