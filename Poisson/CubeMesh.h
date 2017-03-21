#pragma once
class CubeMesh
{
private:
	float* OldData;
	float* NewData;
	int n_rows, n_cols, n_layers, n_elems;
	float hx, hy, hz, h;
	float a, ax, ay, az, ah;
	const float PI = 3.1415927f;
	const float C = -0.008443431966f;
	float error, relative_error, absolute_error;
	

public:
	int* IJK;
	// Constructors
	CubeMesh(int, int, int);							// Constructor, rows by cols by layers mesh. TODO: Decide default values and boundary conditions
	
    // Functions
	float f_ijk(int, int, int);
	float u_exact(int, int, int);
	void ApplyLaplacian();
	void SwapBuffers();
	void ComputeError();
	void ComputeExactError();
	int ijk_to_m(int, int, int);
	void m_to_ijk(int);

	float getError();
	float getRelativeError();
	float getAbsoluteError();
	float get(int, int, int, int);
	void set(float, int, int, int, int);
	void printData(int);

	// Destructor
	//~CubeMesh();	// Free memory upon destruction
};