#include <iostream>
#include <mpi.h>
#include "CubeMesh.h"
using namespace std;

int* get_divs(int);

void Ex2_Parallel(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Initialize local CubeMesh using rank and size
	int* divs = get_divs(size);
	int N = 2;
	CubeMesh cm(N, N, N, rank, size);
	cm.printData(0);
	cm.printData(1);

	// Setup cartesian communicator
	MPI_Comm cart_comm;
	int* periods;
	periods = new int[3]();
	MPI_Cart_create(MPI_COMM_WORLD, 3, divs, periods, 0, &cart_comm);
	// Left/Right datatype
	MPI_Datatype lr;
	MPI_Type_vector(N / divs[1], N / divs[2], N / divs[0], MPI_FLOAT, &lr);
	MPI_Type_commit(&lr);
	// Front/Back datatype
	// Top/Bottom datatype

	MPI_Finalize();
}

int* get_divs(int p)
{
	int p_divs[25][3] = {	{ 1, 1, 1 },
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
							{ 13, 2, 1 }	};
	int* ret;
	ret = new int[3];
	ret[0] = p_divs[p - 1][0];
	ret[1] = p_divs[p - 1][1];
	ret[2] = p_divs[p - 1][2];
	return ret;
}