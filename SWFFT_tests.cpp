#include <iostream>
#include <iomanip>

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "/home/lynchg/SWFFT/complex-type.h"
#include "/home/lynchg/SWFFT/AlignedAllocator.h"
#include "/home/lynchg/SWFFT/Error.h"
#include "/home/lynchg/SWFFT/TimingStats.h"

#include "/home/lynchg/SWFFT/Distribution.hpp"
#ifndef DFFT_TIMING
#define DFFT_TIMING 1
#endif
#include "/home/lynchg/SWFFT/Dfft.hpp"

#define ALIGN 16

using namespace hacc;

uint64_t double_to_uint64_t(double d) {
	uint64_t i;
	memcpy(&i, &d, 8);
	return i;
}//This copies 8 bytes from &d to &i, converting from double to uint64_t (since it returns i)

void assign_delta_function(Dfft &dfft, complex_t *a){
	
	const int *self = dfft.self_rspace();
	//this returns d.process_topology_3.self, which is the coordinates in the process grid 
	const int *local_ng = dfft.local_ng_rspace();
	//this returns d.process_topology_3.n, the local grid dimensions
	//in both cases, 'd' refers to the distribution_t object that the Dfft object is built from

	complex_t zero(0.0, 0.0);
	complex_t one(1.0, 0.0);
	size_t local_indx = 0;
	for(size_t i=0; i<(size_t)local_ng[0]; i++) {
		size_t global_i = local_ng[0]*self[0] + i;
		//x....|.....|..... local_ng=5, self=0, i=0
		//.....|x....|..... local_ng=5, self=1, i=0
		//.x...|.....|..... local_ng=5, self=0, i=1
		
		for(size_t j=0; j<(size_t)local_ng[1]; j++) {
			size_t global_j = local_ng[1]*self[1] + j;

			for(size_t k=0; k<(size_t)local_ng[2]; k++) {
				size_t global_k = local_ng[2]*self[2] + k;

				if(global_i == 0 && global_j == 0 && global_k == 0)
					a[local_indx] = one;
				else
					a[local_indx] = zero;

				local_indx++;
			}
		}
	}
}//for each rank, this assings a global coordinate (in this scope) to each local coordinate
// and if that global coordinate is the origin, it assigns a value of a[origin]=1
// this is the dirac delta function. Local_indx, at the end, should be equal to volume-1

void check_kspace(Dfft &dfft, complex_t *a) {
	
	double LocalRealMin, LocalRealMax, LocalImagMin, LocalImagMax;
	LocalRealMin = LocalRealMax = std::real(a[1]);
	LocalImagMin = LocalImagMax = std::imag(a[1]);
	// these are the local real and imag max mins (within each process block)
	// they are initialize to a[1]

	size_t local_size = dfft.local_size();
	//returns n[0]*n[1]*n[1], or the local volume on the process grid
	
	for(size_t local_indx=0; local_indx<local_size; local_indx++) {
		double re = std::real(a[local_indx]);
		double im = std::imag(a[local_indx]);

		LocalRealMin = re < LocalRealMin ? re : LocalRealMin;
		LocalRealMax = re > LocalRealMax ? re : LocalRealMax;
		LocalImagMin = im < LocalImagMin ? im : LocalImagMin;
		LocalImagMax = im > LocalImagMax ? im : LocalImagMax;
		// condition ? result if true : result if false
	}//this block of code evalutes a for every point in the volume. It compares
	//the real and imaginary bits with the local mins and maxes, and reassings these values
	//if need be.

	const MPI_Comm comm = dfft.parent_comm();//returns Comm used by Dfft object
	double GlobalRealMin, GlobalRealMax, GlobalImagMin, GlobalImagMax;
	MPI_Allreduce(&LocalRealMin, &GlobalRealMin, 1, MPI_DOUBLE, MPI_MIN, comm);
	MPI_Allreduce(&LocalRealMax, &GlobalRealMax, 1, MPI_DOUBLE, MPI_MAX, comm);
	MPI_Allreduce(&LocalImagMin, &GlobalImagMin, 1, MPI_DOUBLE, MPI_MIN, comm);
	MPI_Allreduce(&LocalImagMax, &GlobalImagMax, 1, MPI_DOUBLE, MPI_MAX, comm);
	//(void* send_data, void* recv_data, int count, MPI_Datatype, datatype, MPI_Op opm, Comm)
	//This block communicates the local values of min and max to all ranks and determines the
	//global min and max. It writes this value to &Globalxxxxxxx on each rank. 

	int rank;
	MPI_Comm_rank(comm, &rank);
	if(rank == 0) {
		std::cout << std::endl << "k-space:" << std::endl
				  << "real in " << std::scientific
				  << "[" << GlobalRealMin << "," << GlobalRealMax << "]"
				  << " = " << std::hex
				  << "[" << double_to_uint64_t(GlobalRealMin) << ","
				  << double_to_uint64_t(GlobalRealMax) << "]"
				  << std::endl
				  << "imag in " << std::scientific
				  << "[" << GlobalImagMin << "," << GlobalImagMax << "]"
				  << " = " << std::hex
				  << "[" << double_to_uint64_t(GlobalImagMin) << ","
				  << double_to_uint64_t(GlobalImagMax) << "]"
				  << std::endl << std::endl << std::fixed;
	}
}//this block takes the k space values as input and calculates the global min and maxes for 
//real and imag. It prints the real and imaginary range of values as an int and hex

void check_rspace(Dfft &dfft, complex_t *a){
	
	const int *self = dfft.self_rspace();
  	//coordinates in the process grid

	const int *local_ng = dfft.local_ng_rspace();
	//local dimensions

	double LocalRealMin, LocalRealMax, LocalImagMin, LocalImagMax;
	LocalRealMin = LocalRealMax = std::real(a[1]);
	LocalImagMin = LocalImagMax = std::imag(a[1]);

	const MPI_Comm comm = dfft.parent_comm();
	int rank;
	MPI_Comm_rank(comm, &rank);
	
	if(rank == 0)
		std::cout << std::endl << "r-space:" << std::endl;
  
	size_t local_indx = 0;
	for(size_t i=0; i<(size_t)local_ng[0]; i++) {
		size_t global_i = local_ng[0]*self[0]+i;
			//x....|.....|..... local_ng=5, self=0, 
			//.....|x....|..... local_ng=5, self=1, 
			//.x...|.....|..... local_ng=5, self=0, i=1

		for(size_t j=0; j<(size_t)local_ng[1]; j++) {
			size_t global_j = local_ng[1]*self[1] + j;
			
			for(size_t k=0; k<(size_t)local_ng[2]; k++) {
				size_t global_k = local_ng[2]*self[2] + k;

				if(global_i == 0 && global_j == 0 && global_k == 0) {
					std::cout << "a[0,0,0] = " << std::fixed << a[local_indx]
							  << std::hex << " = ("
							  << double_to_uint64_t(std::real(a[local_indx]))
							  << ","
							  << double_to_uint64_t(std::imag(a[local_indx]))
							  << ")" << std::endl;
				} else {
					double re = std::real(a[local_indx]);
					double im = std::imag(a[local_indx]);
					LocalRealMin = re < LocalRealMin ? re : LocalRealMin;
					LocalRealMax = re > LocalRealMax ? re : LocalRealMax;
					LocalImagMin = im < LocalImagMin ? im : LocalImagMin;
					LocalImagMax = im > LocalImagMax ? im : LocalImagMax;
				}
				
				local_indx++;
			}
		}
	}//this block assigns global coordinates on each rank to each point in 
  	// r space. if that point is the origin, it prints the value of a at that
	// point. Else, it checks if that point should be the new local min or max	

	double GlobalRealMin, GlobalRealMax, GlobalImagMin, GlobalImagMax;
	MPI_Allreduce(&LocalRealMin, &GlobalRealMin, 1, MPI_DOUBLE, MPI_MIN, comm);
	MPI_Allreduce(&LocalRealMax, &GlobalRealMax, 1, MPI_DOUBLE, MPI_MAX, comm);
	MPI_Allreduce(&LocalImagMin, &GlobalImagMin, 1, MPI_DOUBLE, MPI_MIN, comm);
	MPI_Allreduce(&LocalImagMax, &GlobalImagMax, 1, MPI_DOUBLE, MPI_MAX, comm);

	if(rank == 0) {
		std::cout << "real in " << std::scientific
				  << "[" << GlobalRealMin << "," << GlobalRealMax << "]"
				  << " = " << std::hex
				  << "[" << double_to_uint64_t(GlobalRealMin) << ","
				  << double_to_uint64_t(GlobalRealMax) << "]"
				  << std::endl
				  << "imag in " << std::scientific
				  << "[" << GlobalImagMin << "," << GlobalImagMax << "]"
				  << " = " << std::hex
				  << "[" << double_to_uint64_t(GlobalImagMin) << ","
				  << double_to_uint64_t(GlobalImagMax) << "]"
				  << std::endl << std::endl << std::fixed;
	}//this portion sets the global mac and min, and prints those values out
}



void test(MPI_Comm comm, size_t repetitions, int const ng[]){

	Distribution d(comm, ng);
	//Distribution object defined in Distribution.cpp

	Dfft dfft(d);
	//defined in Dfft.cpp

	std::vector<complex_t, AlignedAllocator<complex_t, ALIGN> > a;//what are a,b for?
	std::vector<complex_t, AlignedAllocator<complex_t, ALIGN> > b;//and what is the purpose of the AlignedAllocator?

	size_t local_size = dfft.local_size();
	a.resize(local_size);
	b.resize(local_size);//makes sure arrays can hold all elements from a given proc cube

	dfft.makePlans(&a[0], &b[0], &a[0], &b[0]);
	//forward output, forward scratich, backward input, backward scratch
  	//What are fftw plans? How does fftw work?
	// grab a copy of the 3D cartesian communicator that Distribution is using
	//when you call fft forwrad, a is copied to the scratch, comp performed, and answer
	//writte to a
	MPI_Comm CartComm = d.cart_3d();
  
	int rank;
	MPI_Comm_rank(CartComm, &rank);

	if(rank==0) {
		std::cout << std::endl
				  << "Hex representations of double precision floats"
				  << std::endl;
		double zero = 0.0;
		std::cout << std::scientific << zero << " = " << std::hex
				  << double_to_uint64_t(zero) << std::endl;
		double one = 1.0;
		std::cout << std::scientific << one << " = " << std::hex
				  << double_to_uint64_t(one) << std::endl;
		double Ng = 1.0*(((uint64_t)ng[0])*((uint64_t)ng[1])*((uint64_t)ng[2]));//total number of grid points (from input)
		std::cout << std::fixed << Ng << " = " << std::hex
				  << double_to_uint64_t(Ng) << std::endl;
		std::cout << std::endl;
	}
	
	MPI_Barrier(CartComm);

	for(size_t i=0; i<repetitions; i++) {
		if(rank==0) {
			std::cout << std::endl << "TESTING " << i << std::endl << std::endl;
		}
    
		MPI_Barrier(CartComm);

		double start, stop;

		assign_delta_function(dfft, &a[0]);
		//takes the initial input complex_t array and assigns a delta function to it

		start = MPI_Wtime();
		dfft.forward(&a[0]);
		stop = MPI_Wtime();
		printTimingStats(CartComm, "FORWARD   ", stop-start);
		//this is where the actual fft from real to k space is performed
		//the output is written to a, so the first entry is at mem address &a[0]
		//timing diagnostics are performed
		
		check_kspace(dfft, &a[0]);
		//k space checked, mac and min values of real and imag printed
		
		start = MPI_Wtime();
		dfft.backward(&a[0]);
		stop = MPI_Wtime();
		printTimingStats(CartComm, "BACKWARD  ", stop-start);
		//backwards fft is performed, output also written to a. this should then be in real space

		check_rspace(dfft, &a[0]);
		//max and min values in real space calculacted and value at origin printed
	}
}

int main(int argc, char *argv[]){

	if(argc < 3) {
		std::cerr << "USAGE: " << argv[0] << " <n_repetitions> <ngx> [ngy ngz]" << std::endl;
		return -1;
	}
  
	MPI_Init(&argc, &argv);

	size_t repetitions = atol(argv[1]);
	int ng[3];
	ng[2] = ng[1] = ng[0] = atoi(argv[2]);
	if(argc > 4) {
		ng[1] = atoi(argv[3]);
		ng[2] = atoi(argv[4]);
	}

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // initialize fftw3 openmp threads if necessary
#ifdef _OPENMP
	if(!fftw_init_threads())
		Error() << "fftw_init_threads() failed!";
	int omt = omp_get_max_threads();
	fftw_plan_with_nthreads(omt);
	if(rank==0)
		std::cout << "Threads per process: " << omt << std::endl;
#endif

	test(MPI_COMM_WORLD, repetitions, ng);
  
	MPI_Finalize();
  
	return 0;
}
