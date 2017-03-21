#include "CubeMesh.h"
#include <omp.h>
#include <iostream>
using namespace std;

void Ex1_Sequential()
{
	int N = 30;
	CubeMesh cm(N, N, N);
	cout << "CubeMesh has been initialized.\n";
	cout << "Error after a single initialization iteration: " << cm.getError() << endl;

	/*
	cout << "Printing CubeMesh OldData: \n";
	cm.printData(0);
	cout << "Printing CubeMesh NewData: \n";
	cm.printData(1);
	*/

	double start_time = omp_get_wtime();

	int iteration_counter = 1;
	int max_iterations = 10000;
	float tolerance = 1e-9f;
	float current_error = cm.getError();
	float error_diff = cm.getError();
	while (error_diff > tolerance && iteration_counter <= max_iterations)
	{
		cm.ApplyLaplacian();
		error_diff = fabs(current_error - cm.getError());
		current_error = cm.getError();
		cout << "Error at iteration " << iteration_counter << " is equal to " << error_diff << endl;
		iteration_counter++;
	}
	cout << "Error at iteration " << (iteration_counter - 1) << " is equal to " << current_error << endl;

	/*
	cout << "Printing Final CubeMesh OldData: \n";
	cm.printData(0);
	cout << "Printing Final CubeMesh NewData: \n";
	cm.printData(1);
	*/

	cm.ComputeExactError();
	cout << "Relative Error of final computed solution vs exact solution: " << cm.getRelativeError() << endl;
	cout << "Absolute Error of final computed solution vs exact solution: " << cm.getAbsoluteError() << endl;

	double end_time = omp_get_wtime();

	cout << "Elapsed time: " << end_time - start_time << endl;
}