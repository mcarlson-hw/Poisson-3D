#pragma once
class CubeMesh
{
private:
	// Constants
	const float PI = 3.1415927f;
	const float C = -0.008443431966f;

	// Data
	float* OldData;
	float* NewData;
	float* top_neighbor;
	float* bottom_neighbor;
	float* left_neighbor;
	float* right_neighbor;
	float* front_neighbor;
	float* back_neighbor;

	// Parameters
	int n_rows, n_cols, n_layers, n_elems;
	int local_rank;
	float hx, hy, hz, h;
	float a, ax, ay, az, ah;
	float error, relative_error, absolute_error;
	int* divs;

public:
	int* IJK;

	// Constructors
	CubeMesh(int, int, int);
	CubeMesh(int, int, int, int, int);
	
    // Internal Functions
	void ApplyLaplacian();
	void SwapBuffers();
	void ComputeError();
	void ComputeExactError();

	// Static Functions
	float u_exact(int, int, int);
	float f_ijk(int, int, int);
	void set_divs(int);
	void printData(int);
	
	// Coordinate Functions
	int ijk_to_m(int, int, int);
	int jk_to_m(int, int);
	int ij_to_m(int, int);
	int ik_to_m(int, int);
	void m_to_ijk(int);

	// Accessors
	float getError();
	float getRelativeError();
	float getAbsoluteError();
	float* data_pointer(int);
	
	// Destructor
	//~CubeMesh();	// Free memory upon destruction
};