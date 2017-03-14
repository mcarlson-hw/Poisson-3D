#include <cmath>
#include "CubeMesh.h"
CubeMesh::CubeMesh(int rows, int columns, int layers)
{
	// 1) Initialize single cube mesh of size (rows by columns by layers) with random values in NewData
	//		When ApplyLaplacian is called, NewData will be swapped into OldData, no need to initialize OldData
}

void CubeMesh::ApplyLaplacian()
{
	// Swap NewData and OldData
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
}
void CubeMesh::SwapBuffers()
{

}
int CubeMesh::BoundaryStatus(int i, int j, int k)
{
	return 0;
}
void CubeMesh::ComputeError()
{

}
void CubeMesh::ComputeExactError()
{
	// 5) (Optional) Check against exact solution
}

float CubeMesh::NewData(int i, int j, int k)
{

}
float CubeMesh::OldData(int i, int j, int k)
{

}
float CubeMesh::getError()
{
	return 1.0f;
}
float CubeMesh::getExactError()
{
	return 1.0f;
}