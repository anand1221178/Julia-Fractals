/* File:     fractal.cpp
 *
 * Purpose:  compute the Julia set fractals
 *
 * Compile:  g++ -g -Wall -fopenmp -o fractal fractal.cpp -lglut -lGL
 * Run:      ./fractal
 *
 */

#include <iostream>
#include <cstdlib>
#include "../common/cpu_bitmap.h"
#include <omp.h>
using namespace std;

#define DIM 768 //defines the image dimensions width and height
/*Uncomment the following line for visualization of the bitmap*/
#define NUM_THREADS 8
#define DISPLAY 1


//struct used in the julia function to represent complex numbers on the complex plane
struct cuComplex {
    float   r;
    float   i;
    cuComplex( float a, float b ) : r(a), i(b)  {}
    float magnitude2( void ) { return r * r + i * i; }
    cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

//calculates the membership of a point in the complex plane within the Julia set
//x is the x-coordinate of the pixel in image, y is the y-coordinate of the pixel in image
int julia( int x, int y ) { 
    const float scale = 1.5;
    //calculates sacled versions of x and y -> transforms the pixel coordinates into comlpex plane coordinates suitable for the julia set formula
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8, 0.156);
    // cuComplex c(-0.5, -0.56); //defines object c -> changing this will give us a different julia set
    cuComplex a(jx, jy);// defines object a -> created using the scaled coordinates (jx, jy) asscoiated with the pixel (x, y)

    //iterates a max of 200 times to determine if the point is in the julia set
    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c; //a is squared and added with constant c 
        if (a.magnitude2() > 1000) //squared magnitude of a is compared with 1000 for our divergence check
            return 0; //if the magnitude of a is greater than 1000, the point is not in the julia set
    }

    return 1; //if the point is in the julia set
}

/*Parallelize the following function using OpenMP*/
void kernel_omp_rowwise ( unsigned char *ptr ){
    int nthreads; //used for collection at the end and to set the number of threads in the par region
    int tid, tthreads, y;
    // int juliaValue;
    omp_set_num_threads(NUM_THREADS); //set for 8 for now 768/8 = 96
    #pragma omp parallel private(tid, y)
    {
        tthreads = NUM_THREADS; //get number of threads in the par region
        tid = omp_get_thread_num(); //get the unique thread ids

        if(tid == 0){
            nthreads = tthreads;
        }

        for (y=tid; y < DIM ; y = y + tthreads) //distro over rows here
        {
            int baseOffset = y * DIM;
            for (int x=0; x<DIM; x++)//cols here
            {
                int offset = x + baseOffset; //offset out calculation
                    //gathering all the data now
                int juliaValue = julia( x, y );
                // cout << juliaValue << endl;

                ptr[offset*4 + 0] = 255 * juliaValue;
                ptr[offset*4 + 1] = 0;
                ptr[offset*4 + 2] = 0;
                ptr[offset*4 + 3] = 255;
            }
        }
    }  


 }

void kernal_omp_colwise ( unsigned char *ptr ){
        int nthreads; //used for collection at the end and to set the number of threads in the par region
    int tid, tthreads, x;
    // int juliaValue;
    omp_set_num_threads(NUM_THREADS); //set for 8 for now 768/8 = 96
    #pragma omp parallel private(tid, x)
    {
        tthreads = NUM_THREADS; //get number of threads in the par region
        tid = omp_get_thread_num(); //get the unique thread ids
        //set our master theread responsible for cleanup and distro
        // #pragma omp critical
        if(tid == 0){
            nthreads = tthreads;
        }
        for (x=tid; x < DIM ; x = x + tthreads) //distro over rows here
        {
            int baseOffset = x * DIM;
            for (int y=0; y<DIM; y++)//cols here
            {
                int offset = y + baseOffset; //offset out calculation
                    //gathering all the data now
                int juliaValue = julia( y, x );
                // cout << juliaValue << endl;

                ptr[offset*4 + 0] = 255 * juliaValue;
                ptr[offset*4 + 1] = 0;
                ptr[offset*4 + 2] = 0;
                ptr[offset*4 + 3] = 255;
            }
        }
    }  
 }

 void kernal_omp_rowblock (unsigned char *ptr)
 {
    int tid, rows_per_thread, start_row,end_row,y;
    int remainder = DIM % NUM_THREADS;
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel private(tid, rows_per_thread, start_row, end_row, y)
    {
        tid = omp_get_thread_num();
        int tthreads = omp_get_num_threads();

        rows_per_thread = DIM/tthreads;
        start_row = tid * rows_per_thread;
        end_row = start_row + rows_per_thread -1; 
        //#pragma omp critical
        //cout << "Thread " << tid << " is processing from starting at row: " << start_row << " uptill row: " << end_row << endl;

        if(tid == NUM_THREADS -1 && remainder != 0){
            end_row += remainder;
            //#pragma omp critical
            //cout << " !!!!! Thread " << tid << " is processing from starting at row: " << start_row << " uptill row: " << end_row << endl;
        }

        for(y = start_row; y <= end_row ; y++){
            int baseOffset = y * DIM;
            for(int x = 0; x < DIM ; x++){
                int offset = x + baseOffset; //offset out calculation
                //gathering all the data now
                int juliaValue = julia( x, y );
                // cout << juliaValue << endl;
                ptr[offset*4 + 0] = 255 * juliaValue;
                ptr[offset*4 + 1] = 0;
                ptr[offset*4 + 2] = 0;
                ptr[offset*4 + 3] = 255;
            }
        }
    }
 }

  void kernal_omp_colblock (unsigned char *ptr)
 {
    int tid, cols_per_thread, start_col,end_col,x;
    int remainder = DIM % NUM_THREADS;
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel private(tid, cols_per_thread, start_col, end_col, x)
    {
        tid = omp_get_thread_num();
        int tthreads = omp_get_num_threads();

        cols_per_thread = DIM/tthreads;
        start_col = tid * cols_per_thread;
        end_col = start_col + cols_per_thread -1; 
        //#pragma omp critical
        //cout << "Thread " << tid << " is processing from starting at row: " << start_row << " uptill row: " << end_row << endl;

        if(tid == NUM_THREADS -1 && remainder != 0){
            end_col += remainder;
            //#pragma omp critical
            //cout << " !!!!! Thread " << tid << " is processing from starting at row: " << start_row << " uptill row: " << end_row << endl;
        }

        for(x = start_col; x <= end_col ; x++){
            int baseOffset = x * DIM;
            for(int y = 0; y < DIM ; y++){
                int offset = y + baseOffset; //offset out calculation
                //gathering all the data now
                int juliaValue = julia( y, x );
                // cout << juliaValue << endl;
                ptr[offset*4 + 0] = 255 * juliaValue;
                ptr[offset*4 + 1] = 0;
                ptr[offset*4 + 2] = 0;
                ptr[offset*4 + 3] = 255;
            }
        }
    }
 }
 
 //responsible for calculating and assigning colors to pixels in the image
 void kernel_serial ( unsigned char *ptr ){ //send in a pointer to an array of unsigned chars -> prolly reps image data in memory RGB?
    for (int y=0; y<DIM; y++) { //iterate over the rows of the image
        for (int x=0; x<DIM; x++) { //iterate over the columns of the image
            int offset = x + y * DIM; //used to locate memory location of the pixel in the image data array pointed to by ptr

            int juliaValue = julia( x, y ); //determines if the pixel at (x, y) is in the julia set
            ptr[offset*4 + 0] = 255 * juliaValue; //assigns the color of the pixel based on the juliaValue
            ptr[offset*4 + 1] = 0; //green
            ptr[offset*4 + 2] = 0; //blue
            ptr[offset*4 + 3] = 255; //alpha
        }
    }
 }

int main( void ) {
    CPUBitmap bitmap( DIM, DIM );
    unsigned char *ptr_s = bitmap.get_ptr();
    unsigned char *ptr_p_col = bitmap.get_ptr(); 
    unsigned char *ptr_p_row = bitmap.get_ptr(); 
    unsigned char *ptr_p_2dRow = bitmap.get_ptr();
    unsigned char *ptr_p_2dcol = bitmap.get_ptr();
    double start, finish_s, finish_p_row,finish_p_col, finish_p_2dcol,finish_p_2dRow; 

    start = omp_get_wtime();
    kernel_serial( ptr_s );
	finish_s = omp_get_wtime() - start;
    

   start = omp_get_wtime();
   kernel_omp_rowwise( ptr_p_row );
	finish_p_row = omp_get_wtime() - start;

    start = omp_get_wtime();
    kernal_omp_colwise( ptr_p_col );
	finish_p_col = omp_get_wtime() - start;
    
    start = omp_get_wtime();
    kernal_omp_rowblock( ptr_p_2dRow );
	finish_p_2dRow = omp_get_wtime() - start;

    start = omp_get_wtime();
    kernal_omp_colblock( ptr_p_2dcol );
	finish_p_2dcol = omp_get_wtime() - start;

    cout << "Elapsed time: " << endl;
    cout << "Serial time: " << finish_s << endl;
    cout << "Parallel time row-wise: " << finish_p_row << endl;
    cout << "Speedup row wise: " << finish_s/finish_p_row << endl;
    cout << "Parallel time col-wise: " << finish_p_col << endl;
    cout << "Speedup col wise: " << finish_s/finish_p_col << endl;
    cout << "Parallel time 2drow-wise: " << finish_p_2dRow << endl;
    cout << "Speedup 2drow-wise: " << finish_s/finish_p_2dRow << endl;
    cout << "Parallel time 2dcol-wise: " << finish_p_2dcol << endl;
    cout << "Speedup 2dcol-wise: " << finish_s/finish_p_2dcol << endl;      
	    
    #ifdef DISPLAY     
    bitmap.display_and_exit();
    #endif
}
