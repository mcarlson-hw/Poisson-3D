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
	/*
	for (int i = 0; i < n_rows; i++)
		for (int j = 0; j < n_cols; j++)
			for (int k = 0; k < n_layers; k++)
			{
				sum = 0.0f;
				m = ijk_to_m(i + 1, j, k);
				if (m != -1) { sum += ax*OldData[m]; }
				m = ijk_to_m(i - 1, j, k);
				if (m != -1) { sum += ax*OldData[m]; }
				m = ijk_to_m(i, j + 1, k);
				if (m != -1) { sum += ay*OldData[m]; }
				m = ijk_to_m(i, j - 1, k);
				if (m != -1) { sum += ay*OldData[m]; }
				m = ijk_to_m(i, j, k + 1);
				if (m != -1) { sum += az*OldData[m]; }
				m = ijk_to_m(i, j, k - 1);
				if (m != -1) { sum += az*OldData[m]; }
				m = ijk_to_m(i, j, k);
				NewData[m] = ah*f_ijk(i, j, k) + sum;
			}
	*/
#pragma omp parallel for num_threads(32)
	for (int M = 0; M < n_elems; M++)
	{
		m_to_ijk(M);
		sum = 0.0f;
		m = ijk_to_m(IJK[0] + 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*OldData[m]; }
		m = ijk_to_m(IJK[0] - 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*OldData[m]; }
		m = ijk_to_m(IJK[0], IJK[1] + 1, IJK[2]);
		if (m != -1) { sum += ay*OldData[m]; }
		m = ijk_to_m(IJK[0], IJK[1] - 1, IJK[2]);
		if (m != -1) { sum += ay*OldData[m]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] + 1);
		if (m != -1) { sum += az*OldData[m]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] - 1);
		if (m != -1) { sum += az*OldData[m]; }
		NewData[M] = ah*f_ijk(IJK[0], IJK[1], IJK[2]) + sum;
	}
	ComputeError();
}

float CubeMesh::f_ijk(int i, int j, int k)
{
	return sin(2*PI*(float)(i+1)*hx)*sin(2*PI*(float)(j+1)*hy)*sin(2*PI*(float)(k+1)*hz);
}
float CubeMesh::u_exact(int i, int j, int k)
{
	return C*sin(2 * PI*(float)(i+1)*hx)*sin(2 * PI*(float)(j+1)*hy)*sin(2 * PI*(float)(k+1)*hz);
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
		sum += pow((NewData[i] - OldData[i]),2);
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
void CubeMesh::m_to_ijk(int m)
{
	IJK[0] = m % n_rows;
	IJK[1] = (m / n_rows) % n_cols;
	IJK[2] = m / (n_rows * n_cols);
}
float CubeMesh::get(int i, int j, int k, int b)
{
	if (b == 0) return OldData[ijk_to_m(i, j, k)];
	else if (b == 1) return NewData[ijk_to_m(i, j, k)];
	return NAN;
}
void CubeMesh::set(float v, int i, int j, int k, int b)
{
	if (b == 0) OldData[ijk_to_m(i, j, k)] = v;
	else if (b == 1) NewData[ijk_to_m(i, j, k)] = v;
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