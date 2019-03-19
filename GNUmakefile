# output directory
DFFT_MPI_DIR ?= build

SWFFT_DIR ?= /home/lynchg/SWFFT/build

# MPI C compiler
DFFT_MPI_CC ?= mpicc

# MPI C++ compiler
DFFT_MPI_CXX ?= mpicxx

# pre-processor flags
DFFT_MPI_CPPFLAGS ?= -DDFFT_TIMING=2

# C flags
DFFT_MPI_CFLAGS ?= -g -O3 -Wall -Wno-deprecated -std=gnu99

# C++ flags
DFFT_MPI_CXXFLAGS ?= -g -O3 -Wall -lstdc++

# linker flags
DFFT_MPI_LDFLAGS ?= 

# additional Fortran linker flags
# sometimes this also needs -lmpi++, -lmpicxx, -lmpi_cxx, etc
#DFFT_MPI_FLDFLAGS ?= -lstdc++

# FFTW3
DFFT_FFTW_HOME ?= $(shell dirname$(shell dirname $(shell which fftw-wisdom))) #evaluates to /usr/include
DFFT_FFTW_CPPFLAGS ?= -I$(DFFT_FFTW_HOME)/include
DFFT_FFTW_LDFLAGS ?= -L$(DFFT_FFTW_HOME)/lib -lfftw3 -lm

# these should not usuall require modification
DFFT_MPI_CPPFLAGS += $(DFFT_FFTW_CPPFLAGS)
DFFT_MPI_LDFLAGS += $(DFFT_FFTW_LDFLAGS)

default: nativec utilities

all: nativec utilities fortran

nativec: $(DFFT_MPI_DIR)/SWFFT_tests

utilities: $(DFFT_MPI_DIR)/CheckDecomposition

.PHONY: clean
clean: 
	rm -rf $(DFFT_MPI_DIR) *.mod



$(DFFT_MPI_DIR): 
	mkdir -p $(DFFT_MPI_DIR)

$(DFFT_MPI_DIR)/%.o: %.c | $(DFFT_MPI_DIR)
	$(DFFT_MPI_CC) $(DFFT_MPI_CFLAGS) $(DFFT_MPI_CPPFLAGS) -c -o $@ $<

$(DFFT_MPI_DIR)/%.o: %.cpp | $(DFFT_MPI_DIR)
	$(DFFT_MPI_CXX) $(DFFT_MPI_CXXFLAGS) $(DFFT_MPI_CPPFLAGS) -c -o $@ $<

$(DFFT_MPI_DIR)/test: $(DFFT_MPI_DIR)/SWFFT_tests.o $(SWFFT_DIR)/distribution.o
	$(DFFT_MPI_CXX) $(DFFT_MPI_CXXFLAGS) -o $@ $^ $(DFFT_MPI_LDFLAGS)

$(DFFT_MPI_DIR)/CheckDecomposition: $(SWFFT_DIR)/CheckDecomposition.o $(SWFFT_DIR)/distribution.o
	$(DFFT_MPI_CC) $(DFFT_MPI_CFLAGS) -o $@ $^ $(DFFT_MPI_LDFLAGS)
