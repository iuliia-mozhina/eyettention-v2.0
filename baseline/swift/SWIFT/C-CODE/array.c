
/*
 *	Array utility
 *  Maximilian M. Rabe, 2019
 *  GPL-3 License
 */

#ifndef __SW_ARRAY

#define __SW_ARRAY

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

// Method declarations

static void* ar_sw_array(size_t esize, int ndim, int dims[]) {

	int i,j,k;
	size_t n = 1, cum_n[ndim];

	cum_n[ndim-1] = 0;

	for(i=ndim-1; i>=0; i--) {
		if(i < ndim-1) {
			cum_n[i] = (cum_n[i+1]+1) * dims[i];
		}

		n *= dims[i];
	}

	size_t addr_block_size = sizeof(void*) * (cum_n[0]);
	size_t data_block_size = esize * n;

	size_t block_size = addr_block_size + data_block_size;

	void ** data = (void**) malloc(block_size);

	if(data == NULL) {
		warn("Could not allocate %d-dimensional array with %lu elements and %lu bytes.", ndim, n, block_size);
		return NULL;
	}

	size_t o1 = 0, o2, l =1;
	for(i=0;i<ndim-1;i++) {
		for(j=0;j<l;j++) {
			o2 = o1 + dims[i] * l;
			for(k=0;k<dims[i];k++) {
				size_t from_el = o1+j*dims[i]+k;
				size_t to_el = o2+(j*dims[i]+k)*dims[i+1];
				size_t from_addr, to_addr;
				from_addr = (size_t) &data[from_el];
				if(i == ndim-2) {
					to_addr = (size_t) &data[cum_n[0]] + esize * (to_el - cum_n[0] - 1);
				} else {
					to_addr = (size_t) &data[to_el] - sizeof(void*);
				}
				// printf("Level %d: %d (%lx) points to %d (%lx/%lx).\n", i, from_el, from_addr, to_el, to_addr + sizeof(void*), to_addr + esize);
				*((void**) from_addr) = (void*) to_addr;
			}
		}
		o1 = o2;
		l *= dims[i];
	}
	return (void*) ((size_t) data - (ndim == 1 ? esize : sizeof(void*)));
}

void free_sw_array(size_t esize, void* array, int ndim) {
	if(ndim > 1) {
		free((void*)((size_t) array + sizeof(void*)));
	} else {
		free((void*)((size_t) array + esize));
	}
}

static void ar_sw_resize_array_rec(size_t esize, void* ar1, void* ar2, int ndim, int dims1[], int dims2[]) {
	if(ndim == 1) {
		int n = dims1[0]>dims2[0]?dims2[0]:dims1[0];
		memcpy((void*)((size_t) ar2 + esize), (void*)((size_t) ar1 + esize), n*esize);
	} else {
		int i;
		for(i=1;i<=dims1[0]&&i<=dims2[0];i++) {
			ar_sw_resize_array_rec(esize, ((void**)ar1)[i], ((void**)ar2)[i], ndim-1, &dims1[1], &dims2[1]);
		}
	}
}

static void* ar_sw_resize_array(size_t esize, void* array, int ndim, int dims1[], int dims2[]) {
	void * new_array = ar_sw_array(esize, ndim, dims2);
	ar_sw_resize_array_rec(esize, array, new_array, ndim, dims1, dims2);
	free_sw_array(esize, array, ndim);
	return new_array;
}

static void* va_sw_resize_array(size_t esize, void* array, int ndim, va_list args) {
    int dims1[ndim], dims2[ndim], i;
    int do_resize = 0;
    for(i=0;i<ndim;i++) {
    	dims1[i] = va_arg(args, int);
    }
    for(i=0;i<ndim;i++) {
    	dims2[i] = va_arg(args, int);
    	if(dims2[i]!=dims1[i]) {
    		do_resize = 1;
    	}
    }
    va_end(args);
    if(do_resize) {
		return ar_sw_resize_array(esize, array, ndim, dims1, dims2);
    } else {
    	return array;
    }
}

static void* sw_resize_array(size_t esize, void* array, int ndim, ...) {
	va_list args;
    va_start(args, ndim);
	return va_sw_resize_array(esize, array, ndim, args);
}

#define resize_array(type, ar1, ndim, ...) sw_resize_array(sizeof(type), (void*) ar1, (int) ndim, __VA_ARGS__)

static void* va_sw_array(size_t esize, int ndim, va_list args) {
    int dims[ndim], i;
    for(i=0;i<ndim;i++) {
    	dims[i] = va_arg(args, int);
    	//printf("dim%d: %d\n", i+1, dims[i]);
    }
    va_end(args);
	return ar_sw_array(esize, ndim, dims);
}

void* sw_array(size_t esize, int ndim, ...) {
	va_list args;
    va_start(args, ndim);
    return va_sw_array(esize, ndim, args);
}

static void* ar_sw_array_copy(size_t esize, void * src, void * dest, int ndim, int dims[]) {
	int i,j,k;
	size_t cum_n[ndim], n = 1;

	cum_n[ndim-1] = 0;

	for(i=ndim-1; i>=0; i--) {
		if(i < ndim-1) {
			cum_n[i] = (cum_n[i+1]+1) * dims[i];
		}
		n *= dims[i];
	}

	size_t addr_block_size = sizeof(void*) * cum_n[0];
	size_t data_block_size = n * esize;

	void * real_src, * real_dest;

	if(ndim > 1) {
		real_src = (void*)((size_t) src + sizeof(void*));
		real_dest = (void*)((size_t) dest + sizeof(void*));
	} else {
		real_src = (void*)((size_t) src + esize);
		real_dest = (void*)((size_t) dest + esize);
	}

	memcpy((void*) ((size_t) real_dest + addr_block_size), (void*) ((size_t) real_src + addr_block_size), data_block_size);

	return dest;

}

static void* va_sw_array_copy(size_t esize, void * src, void * dest, int ndim, va_list args) {
    int dims[ndim], i;
    for(i=0;i<ndim;i++) {
    	dims[i] = va_arg(args, int);
    	//printf("dim%d: %d\n", i+1, dims[i]);
    }
    va_end(args);
	return ar_sw_array_copy(esize, src, dest, ndim, dims);
}

void* sw_array_copy(size_t esize, void * src, void * dest, int ndim, ...) {
	va_list args;
    va_start(args, ndim);
    return va_sw_array_copy(esize, src, dest, ndim, args);
}

// Macro definitions

#define array(type, ndim, ...) sw_array(sizeof(type), ndim, __VA_ARGS__)
#define free_array(type, array, ndim) free_sw_array(sizeof(type), array, ndim)

#define copy_array(type, src, dest, ndim, ...) sw_array_copy(sizeof(type), src, dest, ndim, __VA_ARGS__)
#define duplicate_array(type, src, ndim, ...) copy_array(type, src, array(type, ndim, __VA_ARGS__), ndim, __VA_ARGS__)

// Some special cases for matrices

#define imatrix(d1, rows, d2, cols) matrix(int, rows, cols)
#define matrix(type, rows, cols) ((type**) array(type, 2, (int) rows, (int) cols))
#define copy_matrix(type, src, dest, rows, cols) copy_array(type, src, dest, 2, rows, cols)
#define duplicate_matrix(type, src, rows, cols) ((type**) duplicate_array(type, src, 2, rows, cols))
#define free_matrix(type, array) free_array(type, array, 2)
#define dmatrix(d1, rows, d2, cols) matrix(double, rows, cols)
#define free_dmatrix(array, ...) free_matrix(double, (void*) array)
#define free_imatrix(array, ...) free_matrix(int, (void*) array)
#define resize_matrix(type, mat, old_rows, new_rows, old_cols, new_cols) ((type**) resize_array(type, mat, 2, old_rows, old_cols, new_rows, new_cols))

// Some special cases for vectors

#define vector(type, len) ((type*) array(type, 1, (int) len))
#define copy_vector(type, src, dest, len) copy_array(type, src, dest, 1, len)
#define duplicate_vector(type, src, len) ((type*) duplicate_array(type, src, 1, len))
#define free_vector(type, array) free_array(type, array, 1)
#define resize_vector(type, vec, old_len, new_len) ((type*) resize_array(type, vec, 1, old_len, new_len))
#define dvector(d, len) vector(double, len)
#define free_dvector(array, ...) free_vector(double, (void*) array)
#define ivector(d, len) vector(int, len)
#define free_ivector(array, ...) free_vector(int, (void*) array)

// Output macros for matrices and vectors

#define fprintf_matrix(f, mat, rows, cols, format) { \
	size_t i, j;	\
	for(i=1;i<=rows;i++) { \
		fprintf(f, "[%ld/%ld,1..%ld] ", i, (size_t) rows, (size_t) cols); \
		for(j=1;j<=cols;j++) fprintf(f, format, mat[i][j]); \
		fprintf(f, "\n");	\
	}	\
}

#define printf_matrix(...) fprintf_matrix(stdout, __VA_ARGS__)

#define fprintf_vector(f, vec, n, format) {	\
	size_t i, j;	\
	fprintf(f, "[1..%d] ", n);	\
	for(i=1;i<=n;i++) {	\
		fprintf(f, format, vec[i]);	\
	}	\
	fprintf(f, "\n");	\
}

#define printf_vector(...) fprintf_vector(stdout, __VA_ARGS__)


#endif
