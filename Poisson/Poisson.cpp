#include <iostream>
#include "CubeMesh.h"
using namespace std;
int main()
{
	// ##### One Node, Sequential ####
	// Data Structure Notes:
	//	Two 1d Arrays of length rows*columns*layers with float values
	//  Total Memory for a single CubeMesh instance is then 2*rows*columns*layers*4*10E-9 GB
	//  Since a single node on soc-kp has 128 GB of memory available, the maximum size of a CubeMesh is
	//		rows*columns*layers <= 15,983,963,359
	//  If rows = columns = layers = N, then the maximum size of a CubeMesh is
	//						 N	<= 2519
	//
	// 1) Initialize single cube mesh of size (rows by columns by layers) with random values in OldData
	// 2) Initialize NewData with a single step of the Laplacian method:
	//	2a) For i from 1 to rows
	//			For j from 1 to columns
	//				For k from 1 to layers
	//	2b) Check what kind of point (i,j,k) is. Types = {0: interior, 
	//													  1: top, 2: bottom, 3: left, 4: right, 5: front, 6: back,
	//													  7: top-left-front corner, 8: top-left-back corner,
	//													  9: top-right-front corner, 10: top-right-back corner,
	//													  11: bottom-left-front corner, 12: bottom-left-back corner,
	//													  13: bottom-right-front corner, 14: bottom-right-back corner}
	//	2c) If interior point, NewData(i,j,k) = 1/6 * (OldData(i+1,j,k) + OldData(i-1,j,k) + OldData(i,j+1,k)
	//													 + OldData(i,j-1,k) + OldData(i,j,k+1) + OldData(i,j,k-1))
	//	2d) If not interior point, update NewData using available neighbors
	// 3) With NewData and OldData initialized, initialize error value using 2-norm of NewData - OldData,
	//		which can and should be parallelized by OpenMP in the shared memory environment
	// 4) While error value is greater than some tolerance (or a maximum number of iterations),
	//		Swap NewData with OldData, Update NewData by applying Laplacian as outlined above, 
	//		Compute new error value
	// 5) (Optional) Check against exact solution
	// ###############################

	// Step 1, 2, and 3: (rows = columns = layers = N)
	//	First three steps are to initialize an instance of the class and will happen within the constructor of CubeMesh
	int N = 5;
	CubeMesh cm(N, N, N);
	cout << "CubeMesh has been initialized.\n";
	cout << "Error after a single initialization iteration: " << cm.getError() << endl;

	// Step 4: The Loop
	int iteration_counter = 1;
	int max_iterations = 5;
	float tolerance = 1e-5f;
	float current_error = cm.getError();
	while (current_error > tolerance && iteration_counter <= max_iterations)
	{
		cm.ApplyLaplacian();
		cout << "Error at iteration " << iteration_counter << " is equal to " << current_error << endl;
		iteration_counter++;
	}

	// Step 5: Get exact error
	cout << "Relative Error of final computed solution vs exact solution: " << cm.getExactError() << endl;
}