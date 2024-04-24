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

    //cuComplex c(-0.8, 0.156);
    cuComplex c(-0.7269, 0.1889); //defines object c -> changing this will give us a different julia set
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
void kernel_omp ( unsigned char *ptr ){
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia( x, y );
            ptr[offset*4 + 0] = 255 * juliaValue;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
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
    unsigned char *ptr_p = bitmap.get_ptr(); 
    double start, finish_s, finish_p; 
    
    /*Serial run*/
    start = omp_get_wtime();
    kernel_serial( ptr_s );
	finish_s = omp_get_wtime() - start;
    
    /*Parallel run*/ 
    start = omp_get_wtime();
    kernel_omp( ptr_p );
	finish_p = omp_get_wtime() - start;
    
    cout << "Elapsed time: " << endl;
    cout << "Serial time: " << finish_s << endl;
    cout << "Parallel time: " << finish_p << endl;
    cout << "Speedup: " << finish_s/finish_p << endl;
	    
    #ifdef DISPLAY     
    bitmap.display_and_exit();
    #endif
}
