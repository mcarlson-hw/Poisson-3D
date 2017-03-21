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
	
	local_error = 1.0f;

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
CubeMesh::CubeMesh(int total_rows, int total_columns, int total_layers, int rank, int P, int chunk_flag)
{
	set_divs(P);
	p_id = rank;
	n_processors = P;

	int seed = (int)(time(NULL) + rank);
	srand(seed);

	if (chunk_flag == 0)
	{
		n_rows = total_rows / divs[0];
		n_cols = total_columns / divs[1];
		n_layers = total_layers / divs[2];
		n_elems = n_rows*n_cols*n_layers;

		hx = 1.0f / ((float)(total_rows + 1));
		hy = 1.0f / ((float)(total_columns + 1));
		hz = 1.0f / ((float)(total_layers + 1));
	}
	else
	{
		n_rows = total_rows;
		n_cols = total_columns;
		n_layers = total_layers;
		n_elems = n_rows*n_cols*n_layers;

		hx = 1.0f / ((float)(total_rows*divs[0] + 1));
		hy = 1.0f / ((float)(total_columns*divs[1] + 1));
		hz = 1.0f / ((float)(total_layers*divs[2] + 1));
	}


	h = hx*hx * hy*hy * hz*hz;
	a = 1.0f / (hx*hx*hy*hy + hy*hy*hz*hz + hy*hy*hz*hz);
	ah = -0.5f * h * a;
	ax = 0.5f * hy*hy * hz*hz * a;
	ay = 0.5f * hx*hx * hz*hz * a;
	az = 0.5f * hx*hx * hy*hy * a;

	local_error = 1.0f;

	NewData = new float[n_elems];
	OldData = new float[n_elems];
	for (int i = 0; i < n_elems; i++)
		NewData[i] = 2.0f * ((float)rand() / (float)RAND_MAX - 0.5f);
		//NewData[i] = (float)((i+1) + n_elems*rank);

	top_neighbor = new float[n_rows*n_cols]();
	bottom_neighbor = new float[n_rows*n_cols]();
	left_neighbor = new float[n_cols*n_layers]();
	right_neighbor = new float[n_cols*n_layers]();
	front_neighbor = new float[n_rows*n_layers]();
	back_neighbor = new float[n_rows*n_layers]();

	top_data = new float[n_rows*n_cols]();
	bottom_data = new float[n_rows*n_cols]();
	left_data = new float[n_cols*n_layers]();
	right_data = new float[n_cols*n_layers]();
	front_data = new float[n_rows*n_layers]();
	back_data = new float[n_rows*n_layers]();

	IJK = new int[3];
	IJK[0] = -1;
	IJK[1] = -1;
	IJK[2] = -1;
	
	parallel_init();

	ApplyLaplacian();
	MPI_Barrier(MPI_COMM_WORLD);
	ComputeError();
}

void CubeMesh::ApplyLaplacian()
{
	// Swap NewData and OldData
	SwapBuffers();
	PrepareOutgoingBuffers();
	communicate();
	int m;
	float sum;

#pragma omp parallel for
	for (int M = 0; M < n_elems; M++)
	{
		m_to_ijk(M);
		if (IJK[0] == 0 || IJK[0] == n_rows - 1 || IJK[1] == 0 || IJK[0] == n_cols - 1 || IJK[2] == 0 || IJK[2] == n_layers - 1)
			continue;
		sum = 0.0f;
		m = ijk_to_m(IJK[0] + 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*OldData[m]; }
		else { sum += ax*right_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0] - 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*OldData[m]; }
		else { sum += ax*left_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] + 1, IJK[2]);
		if (m != -1) { sum += ay*OldData[m]; }
		else { sum += ay*front_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] - 1, IJK[2]);
		if (m != -1) { sum += ay*OldData[m]; }
		else { sum += ay*back_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] + 1);
		if (m != -1) { sum += az*OldData[m]; }
		else { sum += az*top_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] - 1);
		if (m != -1) { sum += az*OldData[m]; }
		else { sum += az*bottom_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		NewData[M] = ah*f_ijk(IJK[0], IJK[1], IJK[2]) + sum;
	}

	wait_for_recvs();

#pragma omp parallel for
	for (int M = 0; M < n_elems; M++)
	{
		m_to_ijk(M);
		if (IJK[0] != 0 && IJK[0] != n_rows - 1 && IJK[1] != 0 && IJK[0] != n_cols - 1 && IJK[2] != 0 && IJK[2] != n_layers - 1)
			continue;
		sum = 0.0f;
		m = ijk_to_m(IJK[0] + 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*OldData[m]; }
		else { sum += ax*right_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0] - 1, IJK[1], IJK[2]);
		if (m != -1) { sum += ax*OldData[m]; }
		else { sum += ax*left_neighbor[jk_to_m(IJK[1], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] + 1, IJK[2]);
		if (m != -1) { sum += ay*OldData[m]; }
		else { sum += ay*front_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1] - 1, IJK[2]);
		if (m != -1) { sum += ay*OldData[m]; }
		else { sum += ay*back_neighbor[ik_to_m(IJK[0], IJK[2])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] + 1);
		if (m != -1) { sum += az*OldData[m]; }
		else { sum += az*top_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		m = ijk_to_m(IJK[0], IJK[1], IJK[2] - 1);
		if (m != -1) { sum += az*OldData[m]; }
		else { sum += az*bottom_neighbor[ij_to_m(IJK[0], IJK[1])]; }
		NewData[M] = ah*f_ijk(IJK[0], IJK[1], IJK[2]) + sum;
	}

	wait_for_sends();
}
void CubeMesh::SwapBuffers()
{
	float* temp = OldData;
	OldData = NewData;
	NewData = temp;
}
void CubeMesh::PrepareOutgoingBuffers()
{
	for (int j = 0; j < n_cols; j++)
		for (int i = 0; i < n_rows; i++)
		{
			top_data[ij_to_m(i, j)] = OldData[ijk_to_m(i, j, 0)];
			bottom_data[ij_to_m(i, j)] = OldData[ijk_to_m(i, j, n_layers-1)];
		}
	for (int k = 0; k < n_layers; k++)
		for (int i = 0; i < n_rows; i++)
		{
			front_data[ik_to_m(i, k)] = OldData[ijk_to_m(i, n_cols - 1, k)];
			back_data[ik_to_m(i, k)] = OldData[ijk_to_m(i, 0, k)];
		}
	for (int k = 0; k < n_layers; k++)
		for (int j = 0; j < n_cols; j++)
		{
			left_data[jk_to_m(j, k)] = OldData[ijk_to_m(0, j, k)];
			right_data[jk_to_m(j, k)] = OldData[ijk_to_m(n_rows - 1, j, k)];
		}
}
void CubeMesh::ComputeError()
{
	float sum = 0.0f;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < n_elems; i++)
		sum += pow((NewData[i] - OldData[i]), 2);
	local_error = sum;

	MPI_Gather(&local_error, 1, MPI_FLOAT, local_errors, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	float sum2 = 0.0f;
	if (p_id == 0)
	{
		for (int i = 0; i < n_processors; i++)
		{
			sum2 += local_errors[i];
		}
		global_error = sqrt(sum2);
	}
	local_error = sqrt(local_error);

	MPI_Bcast(&global_error, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
}
void CubeMesh::ComputeExactError()
{
	float sum1 = 0.0f;
	float sum2 = 0.0f;
#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < n_elems; i++)
	{
		m_to_ijk(i);
		sum1 += pow((NewData[i] - u_exact(IJK[0],IJK[1],IJK[2])), 2);
		sum2 += pow(u_exact(IJK[0], IJK[1], IJK[2]), 2);
	}
	local_abserror = sum1;
	local_exerror = sum2;

	MPI_Gather(&local_abserror, 1, MPI_FLOAT, local_abserrors, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Gather(&local_exerror, 1, MPI_FLOAT, local_exerrors, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	if (p_id == 0)
	{
		float sum3 = 0.0f;
		float sum4 = 0.0f;
		for (int i = 0; i < n_processors; i++)
		{
			sum3 += local_abserrors[i];
			sum4 += local_exerrors[i];
		}
		global_abserror = sqrt(sum3);
		global_exerror = sqrt(sum4);
	}
	local_abserror = sqrt(local_abserror);
	local_exerror = sqrt(local_exerror);

	MPI_Bcast(&global_abserror, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&global_exerror, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

float CubeMesh::f_ijk(int i, int j, int k)
{
	int I = (i + 1) + (n_rows * p_XYZ[0]);
	int J = (j + 1) + (n_cols * p_XYZ[1]);
	int K = (k + 1) + (n_layers * p_XYZ[2]);
	float ret = sin(2 * PI*(float)I*hx)*sin(2 * PI*(float)J*hy)*sin(2 * PI*(float)K*hz);
	return ret;
}
float CubeMesh::u_exact(int i, int j, int k)
{
	int I = (i + 1) + (n_rows * p_XYZ[0]);
	int J = (j + 1) + (n_cols * p_XYZ[1]);
	int K = (k + 1) + (n_layers * p_XYZ[2]);
	float ret = C*sin(2 * PI*(float)I*hx)*sin(2 * PI*(float)J*hy)*sin(2 * PI*(float)K*hz);
	return ret;
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
	std::cout << "P (" << p_id << "): [";
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

float CubeMesh::getLocalError()
{
	return local_error;
}
float CubeMesh::getGlobalError()
{
	return global_error;
}
float CubeMesh::getGlobalAbsoluteError()
{
	return global_abserror;
}
float CubeMesh::getGlobalRelativeError()
{
	return global_abserror / global_exerror;
}
float* CubeMesh::data_pointer(int b_id, int dir)
{
	if (dir == 0)
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
	else if (dir == 1)
		switch (b_id)
		{
		case 0:	return OldData;
		case 1: return top_data;
		case 2: return bottom_data;
		case 3: return left_data;
		case 4: return right_data;
		case 5: return front_data;
		case 6: return back_data;
		default: { std::cout << "Invalid boundary id.\n"; return NULL; }
		}
	std::cout << "Invalid dir.\n";
	return NULL;
}

void CubeMesh::parallel_init()
{
	if (p_id == 0)
	{
		local_errors = new float[n_processors]();
		local_abserrors = new float[n_processors]();
		local_exerrors = new float[n_processors]();
	}

	MPI_Cart_create(MPI_COMM_WORLD, 3, divs, periods, 0, &cart_comm);
	int local_coords[3] = { 0, 0, 0 };
	MPI_Cart_coords(cart_comm, p_id, 3, local_coords);
	p_XYZ = new int[3];
	p_XYZ[0] = local_coords[0];
	p_XYZ[1] = local_coords[1];
	p_XYZ[2] = local_coords[2];

	p_up = -1;
	p_down = -1;
	p_left = -1;
	p_right = -1;
	p_front = -1;
	p_back = -1;

	int right[3] =	{ local_coords[0] + 1, local_coords[1], local_coords[2] };
	int left[3] =	{ local_coords[0] - 1, local_coords[1], local_coords[2] };
	int front[3] =  { local_coords[0], local_coords[1] + 1, local_coords[2] };
	int back[3] =   { local_coords[0], local_coords[1] - 1, local_coords[2] };
	int up[3] =     { local_coords[0], local_coords[1], local_coords[2] + 1 };
	int down[3] =   { local_coords[0], local_coords[1], local_coords[2] - 1 };

	if (local_coords[0] + 1 < divs[0]) MPI_Cart_rank(cart_comm, right, &p_right);
	if (local_coords[0] - 1 >= 0) MPI_Cart_rank(cart_comm, left, &p_left);
	if (local_coords[1] + 1 < divs[1]) MPI_Cart_rank(cart_comm, front, &p_front);
	if (local_coords[1] - 1 >= 0) MPI_Cart_rank(cart_comm, back, &p_back);
	if (local_coords[2] + 1 < divs[2]) MPI_Cart_rank(cart_comm, up, &p_up);
	if (local_coords[2] - 1 >= 0) MPI_Cart_rank(cart_comm, down, &p_down);
}
void CubeMesh::communicate()
{
	if (p_up != -1)
	{
		MPI_Isend(top_data, n_rows*n_cols, MPI_FLOAT, p_up, 0, cart_comm, &up_s);
		MPI_Irecv(top_neighbor, n_rows*n_cols, MPI_FLOAT, p_up, 0, cart_comm, &up_r);
	}
	if (p_down != -1)
	{
		MPI_Isend(bottom_data, n_rows*n_cols, MPI_FLOAT, p_down, 0, cart_comm, &down_s);
		MPI_Irecv(bottom_neighbor, n_rows*n_cols, MPI_FLOAT, p_down, 0, cart_comm, &down_r);
	}
	if (p_left != -1)
	{
		MPI_Isend(left_data, n_cols*n_layers, MPI_FLOAT, p_left, 0, cart_comm, &left_s);
		MPI_Irecv(left_neighbor, n_cols*n_layers, MPI_FLOAT, p_left, 0, cart_comm, &left_r);
	}
	if (p_right != -1)
	{
		MPI_Isend(right_data, n_cols*n_layers, MPI_FLOAT, p_right, 0, cart_comm, &right_s);
		MPI_Irecv(right_neighbor, n_cols*n_layers, MPI_FLOAT, p_right, 0, cart_comm, &right_r);
	}
	if (p_front != -1)
	{
		MPI_Isend(front_data, n_rows*n_layers, MPI_FLOAT, p_front, 0, cart_comm, &front_s);
		MPI_Irecv(front_neighbor, n_rows*n_layers, MPI_FLOAT, p_front, 0, cart_comm, &front_r);
	}
	if (p_back != -1)
	{
		MPI_Isend(back_data, n_rows*n_layers, MPI_FLOAT, p_back, 0, cart_comm, &back_s);
		MPI_Irecv(back_neighbor, n_rows*n_layers, MPI_FLOAT, p_back, 0, cart_comm, &back_r);
	}
}

void CubeMesh::wait_for_sends()
{
	if (up_s != NULL) MPI_Wait(&up_s, MPI_STATUS_IGNORE);
	if (down_s != NULL) MPI_Wait(&down_s, MPI_STATUS_IGNORE);
	if (left_s != NULL) MPI_Wait(&left_s, MPI_STATUS_IGNORE);
	if (right_s != NULL) MPI_Wait(&right_s, MPI_STATUS_IGNORE);
	if (front_s != NULL) MPI_Wait(&front_s, MPI_STATUS_IGNORE);
	if (back_s != NULL) MPI_Wait(&back_s, MPI_STATUS_IGNORE);
}
void CubeMesh::wait_for_recvs()
{
	if (up_r != NULL) MPI_Wait(&up_r, MPI_STATUS_IGNORE);
	if (down_r != NULL) MPI_Wait(&down_r, MPI_STATUS_IGNORE);
	if (left_r != NULL) MPI_Wait(&left_r, MPI_STATUS_IGNORE);
	if (right_r != NULL) MPI_Wait(&right_r, MPI_STATUS_IGNORE);
	if (front_r != NULL) MPI_Wait(&front_r, MPI_STATUS_IGNORE);
	if (back_r != NULL) MPI_Wait(&back_r, MPI_STATUS_IGNORE);
}