#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#include "mmio.h"

//define structs for CSC and COO format
struct CSCMatrix{
     int *rows;
     int *cols_ptrs;
     int *values;
     int size;
     int nz;
};


struct COOMatrix{
     int *I;
     int *J;
     int *V;
     int size;
     int nz;
};

//declare global pointers for the matrix we'll use
struct COOMatrix *A;
struct CSCMatrix *M;
float *final_arr;


//mergesort algorithm to sort COO matrix before converting it to CSC
void merge(struct COOMatrix *M,int l, int m, int r){
     int i,j,k;
     int n1 = m-l+1;
     int n2 = r-m;
     int *I_L,*J_L,*I_R,*J_R;
     I_L = malloc(n1*sizeof(int));
     J_L = malloc(n1*sizeof(int));
     I_R = malloc(n2*sizeof(int));
     J_R = malloc(n2*sizeof(int));

     for(i=0;i<n1;i++){
          I_L[i] = M->I[i+l];
          J_L[i] = M->J[i+l];
     }
     for(j=0;j<n2;j++){
          I_R[j] = M->I[m+j+1];
          J_R[j] = M->J[m+j+1];
     }

     i=0;
     j=0;
     k=l;
     while(i<n1 && j<n2){
          if(J_L[i]<J_R[j] || (J_L[i]==J_R[j] && I_L[i]<I_R[j])){
               M->I[k] = I_L[i];
               M->J[k] = J_L[i];
               i++;
          }
          else{
               M->I[k] = I_R[j];
               M->J[k] = J_R[j];
               j++;
          }
          k++;
     }
     while(i<n1){
          M->I[k] = I_L[i];
          M->J[k] = J_L[i];
          i++;
          k++;
     }
     while(j<n2){
          M->I[k] = I_R[j];
          M->J[k] = J_R[j];
          j++;
          k++;
     }
     free (I_L);
     free (I_R);
     free (J_L);
     free (J_R);
}

void merge_sortMatrix(struct COOMatrix *M, int l, int r){
     int m;
     if (l<r){
          m = (l+r)/2;
          merge_sortMatrix(M,l,m);
          merge_sortMatrix(M,m+1,r);
          merge(M,l,m,r);
     }
}

//COOtoCSC() function creates a CSC matrix from a sorted CSR matrix and store them in a struct
void COOtoCSC(struct COOMatrix *A){
     int i;
     merge_sortMatrix(A,0,A->nz-1);
     M = malloc(sizeof(struct CSCMatrix));
     int *rows = calloc(A->nz,sizeof(int));
     int *cols_ptrs = calloc(A->size+1,sizeof(int));
     int *values = calloc(A->nz,sizeof(int));
     int size = A->size;
     int nz = A->nz;
     for (i=0;i<nz;i++){
          values[i] = A->V[i];
          rows[i] = A->I[i];
          cols_ptrs[A->J[i]+1]++;
     }
     for (i=0;i<size;i++){
          cols_ptrs[i+1] += cols_ptrs[i];
     }
     M->rows = rows;
     M->cols_ptrs = cols_ptrs;
     M->values = values;
     M->size = size;
     M->nz = nz;
}

//make C00 matrix (code modified from example_read.c)
void makeMatrix(char* name){
     int ret_code;     
     MM_typecode matcode;
     FILE *f;
     int M,N, nz;
     int i;
     int *I, *J, *V;

     A = malloc(sizeof(struct COOMatrix));
     if ((f = fopen(name, "r")) == NULL)
          exit(1);
     if (mm_read_banner(f, &matcode) != 0){
          printf("Can't process Matrix Market banner. \n");
          exit(1);
     }
     if ((ret_code = mm_read_mtx_crd_size(f,&M,&N,&nz)) != 0)
          exit(1);
     //save all coordinates of nonzero elements and not just the upper diagonal
     I = (int*) malloc(2*nz*sizeof(int));
     J = (int*) malloc(2*nz*sizeof(int));
     V = (int*)malloc(2*nz*sizeof(int));
     for (i=0;i<nz;i++){
          fscanf(f,"%d %d \n", &I[i], &J[i]);
          I[i]--;
          J[i]--;
          V[i] = 0;
          I[nz+i] = J[i];
          J[nz+i] = I[i];
          V[nz+i] = 0;
     }
     A->I = I;
     A->J = J;
     A->V = V;
     A->size = M;
     A->nz =2*nz;
     if(f != stdin) fclose(f);
}

//parallelized omp version of sequential_triangle_counting(), outer loop is parallelized with dynamic schedule
void omp_triangle_counting(){
     #pragma omp parallel for schedule(dynamic) 
     for(int i=0;i<M->size;i++){
          for (int j=M->cols_ptrs[i];j<M->cols_ptrs[i+1];j++){
               M->values[j] = vectorsmultiply(i,M->rows[j]);
          }
     }      
}

//same function as vector2vectorMult() in julia
int vectorsmultiply(int row,int col){
     int row_start,row_end,col_start,col_end,tmp_i,tmp_j,sum;
     row_start = M->cols_ptrs[row];
     row_end = M->cols_ptrs[row+1];
     col_start = M->cols_ptrs[col];
     col_end = M->cols_ptrs[col+1];
     tmp_i = col_start;
     tmp_j = row_start;
     sum = 0;
     while (tmp_i<col_end && tmp_j<row_end){
          if (M->rows[tmp_i] < M->rows[tmp_j]){
               tmp_i++;
          }
          else if (M->rows[tmp_i] > M->rows[tmp_j]){
               tmp_j++;
          }
          else{
               sum ++;
               tmp_i++;
               tmp_j++;
          }
     }
     return sum;
}
//multiply matrix by a Mx1 vector of 1/2
float* dimReduce(){
     int i,j,col,col_start,col_end,sum;
     float* final_arr = calloc(M->size,sizeof(int));
     //loop through the columns, sum the rows of each column and divide by 2
     for (i=0;i<M->size;i++){
          col = i;
          col_start = M->cols_ptrs[col];
          col_end = M->cols_ptrs[col+1];
          sum = 0;
          for (j=col_start;j<col_end;j++){
               sum += M->values[j];
          }
          final_arr[col] = sum/2;

     }
     return final_arr;
}

//sum values of final array and divide by 3 to get the final result
int triangle_count(float* arr,int size){
     int i;
     int num=0;
     for (i=0;i<size;i++){
          num += arr[i];
     }
     return num/3;
}

int main(int argc, char *argv[])
{
     struct timeval t_start,t_end;
     double exec_time;
     char* name;
     int i;
      //we expect as argument the name of the file
     if (argc < 2){
          exit(1);
     }
     else{
          //make a COO matrix from the file we specified in the argument
          name = argv[1];
          makeMatrix(name);
     }
     //sort it and convert it to CSC
     COOtoCSC(A);

     gettimeofday(&t_start,NULL);
     omp_triangle_counting();
     gettimeofday(&t_end,NULL);

     free(A);
     
     
     float *final_arr = dimReduce();

     int final = triangle_count(final_arr,M->size);
     //find the execution time of sequential_triangle_counting()
     exec_time = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
     exec_time += (t_end.tv_usec - t_start.tv_usec) / 1000.0;

     printf("Triangles: %d Exec time; %.3lf s\n",final,exec_time/1000);
     free(M);
     free(final_arr);

     return 0;
}

