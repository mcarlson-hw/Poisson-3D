#include <iostream>
#include <omp.h>
#include "CubeMesh.h"
using namespace std;
int main(int argc, char **argv)
{
	
	MPI_Init(&argc, &argv);
	MPI_Barrier(MPI_COMM_WORLD);
	double start_time = omp_get_wtime();

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int N = 20;
	CubeMesh cm(N, N, N, rank, size, 0);	// Strong scaling
	//CubeMesh cm(N, N, N, rank, size, 1);	// Weak scaling

	int iteration_counter = 1;
	int max_iterations = 10000;
	float tolerance = 1e-9f;
	float current_error = cm.getGlobalError();
	float error_diff = cm.getGlobalError();
	while (error_diff > tolerance && iteration_counter <= max_iterations)
	{
		cm.ApplyLaplacian();
		MPI_Barrier(MPI_COMM_WORLD);
		cm.ComputeError();
		error_diff = fabs(current_error - cm.getGlobalError());
		current_error = cm.getGlobalError();
		//if (rank == 0)
			//cout << "Error difference at iteration " << iteration_counter << " is equal to " << error_diff << endl;
		iteration_counter++;
	}
	if (rank == 0)
		cout << "Error at iteration " << (iteration_counter - 1) << " is equal to " << current_error << endl;
	cm.ComputeExactError();
	if (rank == 0)
	{
		cout << "Absolute error at iteration " << (iteration_counter - 1) << " is equal to " << cm.getGlobalAbsoluteError() << endl;
		cout << "Relative error at iteration " << (iteration_counter - 1) << " is equal to " << cm.getGlobalRelativeError() << endl;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double end_time = omp_get_wtime();

	if (rank == 0)
		cout << "Time elapsed: " << end_time - start_time << endl;

	MPI_Finalize();
}