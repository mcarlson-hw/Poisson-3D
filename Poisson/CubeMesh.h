#pragma once
#include <mpi.h>
class CubeMesh
{
private:
	// Constants
	const float PI = 3.1415927f;
	const float C = -0.008443431966f;
	const int periods[3] = { 0, 0, 0 };

	// Data
	float* OldData;
	float* NewData;
	float* top_neighbor;
	float* bottom_neighbor;
	float* left_neighbor;
	float* right_neighbor;
	float* front_neighbor;
	float* back_neighbor;
	float* top_data;
	float* bottom_data;
	float* left_data;
	float* right_data;
	float* front_data;
	float* back_data;

	// Parameters
	int n_rows, n_cols, n_layers, n_elems;
	
	float hx, hy, hz, h;
	float a, ax, ay, az, ah;
	float local_error, global_error;
	float local_abserror, global_abserror;
	float local_exerror, global_exerror;
	float* local_errors;
	float* local_abserrors;
	float* local_exerrors;
	int* divs;

	// MPI Stuff
	int p_id, n_processors, cart_rank;
	MPI_Comm cart_comm;
	int p_up, p_down, p_left, p_right, p_front, p_back;
	int* p_XYZ;
	MPI_Request up_r, down_r, left_r, right_r, front_r, back_r;
	MPI_Request up_s, down_s, left_s, right_s, front_s, back_s;

public:
	int* IJK;
	

	// Constructors
	CubeMesh(int, int, int);
	CubeMesh(int, int, int, int, int, int);
	
    // Internal Functions
	void ApplyLaplacian();
	void SwapBuffers();
	void PrepareOutgoingBuffers();
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
	float getLocalError();
	float getGlobalError();
	float getGlobalAbsoluteError();
	float getGlobalRelativeError();
	float* data_pointer(int, int);

	// MPI
	void parallel_init();
	void communicate();
	void wait_for_sends();
	void wait_for_recvs();
	
	// Destructor
	//~CubeMesh();	// Free memory upon destruction
};