#pragma once
class CubeMesh
{
private:
	float* InteriorData, OldInteriorData;												// Self Interior Data, length = (n_rows-1) by (n_cols-1) by (n_layers-1)
	float* TopData, BottomData, LeftData, RightData, FrontData, BackData;				// Self Boundary Data
	float* n_TopData, n_BottomData, n_LeftData, n_RightData, n_FrontData, n_BackData;	// Neighbor Boundary Data (if p == 1, these are dirichlet boundary conditions)
	float* BoundaryDataPointers, n_BoundaryDataPointers;								// Pointers to Boundary Data for use with MPI communication
	int n_rows, n_cols, n_layers;														// Total number of points in each dimension. (rows <-> x, cols <-> y, layers <-> z)
	float error;
public:
	// Constructors
	CubeMesh(int, int, int);							// Constructor, rows by cols by layers mesh. TODO: Decide default values and boundary conditions
														//CubeMesh(int, int, int, float(*f)(int, int, int));	// Constructor, rows by cols by layers mesh. Data(i,j,k) = f(i, j, k)

														// Operator Overrides

														// Functions
	void ApplyLaplacian();				// Apply laplacian to data
	void SwapBuffers();
	void ComputeError();
	void ComputeExactError();
	int BoundaryStatus(int, int, int);	// Returns {0: interior, 1: top, 2: bottom, 3: left, 4: right, 5: front, 6: back}

										// Getters/Setters
	float NewData(int, int, int);
	float OldData(int, int, int);
	float getError();
	float getExactError();

	// Destructor
	//~CubeMesh();	// Free memory upon destruction
};