#include <cmath>
#include <iostream>
#include <time.h>
#include "CubeMesh.h"
#include <omp.h>

CubeMesh::CubeMesh(int rows, int columns, int layers)
{
	// 1) Initialize single cube mesh of size (rows by columns by layers) with random values in NewData
	//		When ApplyLaplacian is called, NewData will be swapped into OldData, no need to initialize OldData

	int seed = (int)time(NULL);
	srand(seed);

	n_rows = rows;
	n_cols = columns;
	n_layers = layers;
	n_elems = rows*columns*layers;

	hx = 1.0f / ((float)(rows+1));
	hy = 1.0f / ((float)(columns+1));
	hz = 1.0f / ((float)(layers+1));
	h = hx*hx * hy*hy * hz*hz;
	a = 1.0f / (hx*hx*hy*hy + hy*hy*hz*hz + hy*hy*hz*hz);
	ah = -0.5f * h * a;
	ax =  0.5f * hy*hy * hz*hz * a;
	ay =  0.5f * hx*hx * hz*hz * a;
	az =  0.5f * hx*hx * hy*hy * a;
	
	error = 1.0f;

	NewData = new float[n_elems];
	OldData = new float[n_elems];
	for (int i = 0; i < n_elems; i++)
		NewData[i] = 2.0f * ((float)rand() / (float)RAND_MAX - 0.5f);
	
	top_neighbor = new float[n_rows*n_cols]();
	bottom_neighbor = new float[n_rows*n_cols]();
	left_neighbor = new float[n_cols*n_layers]();
	right_neighbor = new float[n_cols*n_layers]();
	front_neighbor = new float[n_rows*n_layers]();
	back_neighbor = new float[n_rows*n_layers]();

	IJK = new int[3];
	IJK[0] = -1;
	IJK[1] = -1;
	IJK[2] = -1;

	ApplyLaplacian();
}
CubeMesh::CubeMesh(int total_rows, int total_columns, int total_layers, int rank, int P)
{
	std::cout << "Beginning CubeMesh Initialization on processor " << rank + 1 << std::endl;
	set_divs(P);
	local_rank = rank;

	int seed = (int)(time(NULL) + rank);
	srand(seed);

	std::cout << "X Divs: " << divs[0] << ", Y Divs: " << divs[1] << ", Z Divs: " << divs[2] << std::endl;

	n_rows = total_rows / divs[0];
	n_cols = total_columns / divs[1];
	n_layers = total_layers / divs[2];
	n_elems = n_rows*n_cols*n_layers;

	std::cout << "Rows: " << n_rows << ", Columns: " << n_cols << ", Layers: " << n_layers << std::endl;

	hx = 1.0f / ((float)(total_rows + 1));
	hy = 1.0f / ((float)(total_columns + 1));
	hz = 1.0f / ((float)(total_layers + 1));
	h = hx*hx * hy*hy * hz*hz;
	a = 1.0f / (hx*hx*hy*hy + hy*hy*hz*hz + hy*hy*hz*hz);
	ah = -0.5f * h * a;
	ax = 0.5f * hy*hy * hz*hz * a;
	ay = 0.5f * hx*hx * hz*hz * a;
	az = 0.5f * hx*hx * hy*hy * a;

	error = 1.0f;

	NewData = new float[n_elems];
	OldData = new float[n_elems];
	for (int i = 0; i < n_elems; i++)
		NewData[i] = 2.0f * ((float)rand() / (float)RAND_MAX - 0.5f);
		//NewData[i] = (float)(i+1);

	top_neighbor = new float[n_rows*n_cols]();
	bottom_neighbor = new float[n_rows*n_cols]();
	left_neighbor = new float[n_cols*n_layers]();
	right_neighbor = new float[n_cols*n_layers]();
	front_neighbor = new float[n_rows*n_layers]();
	back_neighbor = new float[n_rows*n_layers]();

	IJK = new int[3];
	IJK[0] = -1;
	IJK[1] = -1;
	IJK[2] = -1;

	ApplyLaplacian();
}

void CubeMesh::ApplyLaplacian()
{
	// Swap NewData and OldData
	SwapBuffers();

	int m;
	float sum;

#pragma omp parallel for
	for (int M = 0; M < n_elems; M++)
	{
		m_to_ijk(M);
		sum = 0.0f;
		m = ijk_to_m(IJK[0] + 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*OldData[m]; }
		else { sum += right_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0] - 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*OldData[m]; }
		else { sum += left_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] + 1, IJK[2]);
		if (m != -1) { sum += ay*OldData[m]; }
		else { sum += front_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] - 1, IJK[2]);
		if (m != -1) { sum += ay*OldData[m]; }
		else { sum += back_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] + 1);
		if (m != -1) { sum += az*OldData[m]; }
		else { sum += top_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] - 1);
		if (m != -1) { sum += az*OldData[m]; }
		else { sum += bottom_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		NewData[M] = ah*f_ijk(IJK[0], IJK[1], IJK[2]) + sum;
	}

	ComputeError();
}
void CubeMesh::SwapBuffers()
{
	float* temp = OldData;
	OldData = NewData;
	NewData = temp;
}
void CubeMesh::ComputeError()
{
	float sum = 0.0f;
#pragma omp parallel for reduction(+:sum) num_threads(32)
	for (int i = 0; i < n_elems; i++)
		sum += pow((NewData[i] - OldData[i]), 2);
	error = sqrt(sum);
}
void CubeMesh::ComputeExactError()
{
	float sum1 = 0.0f;
	float sum2 = 0.0f;
	for (int i = 0; i < n_rows; i++)
		for (int j = 0; j < n_cols; j++)
			for (int k = 0; k < n_layers; k++)
			{
				sum1 += pow(NewData[ijk_to_m(i, j, k)] - u_exact(i, j, k), 2);
				sum2 += pow(u_exact(i, j, k), 2);
			}
	absolute_error = sqrt(sum1);
	relative_error = absolute_error / sqrt(sum2);
}

float CubeMesh::f_ijk(int i, int j, int k)
{
	return sin(2*PI*(float)(i+1)*hx)*sin(2*PI*(float)(j+1)*hy)*sin(2*PI*(float)(k+1)*hz);
}
float CubeMesh::u_exact(int i, int j, int k)
{
	return C*sin(2 * PI*(float)(i+1)*hx)*sin(2 * PI*(float)(j+1)*hy)*sin(2 * PI*(float)(k+1)*hz);
}
void CubeMesh::set_divs(int p)
{
	int p_divs[25][3] = { { 1, 1, 1 },
	{ 2, 1, 1 },
	{ 3, 1, 1 },
	{ 2, 2, 1 },
	{ 5, 1, 1 },
	{ 3, 2, 1 },
	{ 7, 1, 1 },
	{ 2, 2, 2 },
	{ 3, 3, 1 },
	{ 5, 2, 1 },
	{ 11, 1, 1 },
	{ 3, 2, 2 },
	{ 13, 1, 1 },
	{ 7, 2, 1 },
	{ 5, 3, 1 },
	{ 4, 2, 2 },
	{ 17, 1, 1 },
	{ 3, 3, 2 },
	{ 5, 2, 2 },
	{ 7, 3, 1 },
	{ 11, 2, 1 },
	{ 23, 1, 1 },
	{ 4, 3, 2 },
	{ 5, 5, 1 },
	{ 13, 2, 1 } };
	
	this->divs = new int[3];
	this->divs[0] = p_divs[p - 1][0];
	this->divs[1] = p_divs[p - 1][1];
	this->divs[2] = p_divs[p - 1][2];
}
void CubeMesh::printData(int b)
{
	std::cout << "[";
	if (b == 0)
		for (int i = 0; i < n_elems; i++)
			std::cout << OldData[i] << " ";
	else if (b == 1)
		for (int i = 0; i < n_elems; i++)
			std::cout << NewData[i] << " ";
	std::cout << "]\n";
}

int CubeMesh::ijk_to_m(int i, int j, int k)
{
	if (i < 0 || i > n_rows - 1 || j < 0 || j > n_cols - 1 || k < 0 || k > n_layers - 1)
		return -1;
	return (i + j * n_rows + k * n_rows * n_cols);
}
int CubeMesh::jk_to_m(int j, int k)
{
	// n_cols by n_layers
	return j + k*n_cols;
}
int CubeMesh::ij_to_m(int i, int j)
{
	// n_rows by n_cols
	return i + j*n_rows;
}
int CubeMesh::ik_to_m(int i, int k)
{
	// n_rows by n_layers
	return i + k*n_rows;
}
void CubeMesh::m_to_ijk(int m)
{
	IJK[0] = m % n_rows;
	IJK[1] = (m / n_rows) % n_cols;
	IJK[2] = m / (n_rows * n_cols);
}

float CubeMesh::getError()
{
	return error;
}
float CubeMesh::getRelativeError()
{
	return relative_error;
}
float CubeMesh::getAbsoluteError()
{
	return absolute_error;
}
float* CubeMesh::data_pointer(int b_id)
{
	switch (b_id)
	{
	case 0:	return OldData;
	case 1: return top_neighbor;
	case 2: return bottom_neighbor;
	case 3: return left_neighbor;
	case 4: return right_neighbor;
	case 5: return front_neighbor;
	case 6: return back_neighbor;
	default: { std::cout << "Invalid boundary id.\n"; return NULL; }
	}
}