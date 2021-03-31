#include<stdio.h>
#include<mpi.h>
#include<math.h>
#include<stdlib.h>
#include<string>
#include<sstream>
#include<fstream>

double **a;
double **b;
int n = 3;
int m = 4;
double *pivotrow;

int malloc2ddouble(double ***array, int n, int m) 
{
 double *p = (double *)calloc(n*m,sizeof(double));
   
 if (!p) return -1;

 (*array) = (double **)calloc(n,sizeof(double*));

 if (!(*array)) 
 {
  free(p);

  return -1;
 }

 for (int i=0; i<n; i++) 
 (*array)[i] = &(p[i*m]);

 return 0;
}

int free2ddouble(double ***array) 
{
 /* free the memory - the first element of the array is at the start */
 free(&((*array)[0][0]));

 /* free the pointers into the memory */
 free(*array);

 return 0;
}

void broadcat(double** b,double* pivot,int k,int rank_ystart)
{
 *pivot = b[k][k+rank_ystart];

 for(int s=0; s<m; s++)
 	pivotrow[s] = b[k][s];
}

void gauss_ele(double** b,double* pivot,int k,int i,int rank,int rank_ny,int* rank_ystart0)
{
 if(rank ==i)
 {
  int z= k+1;

  while( z<rank_ny )
  {
   for(int j=m-1; j >= 0; j--)
   {
    b[z][j] = b[z][j] - b[z][k+rank_ystart0[i]]*pivotrow[j]/(*pivot);
   }
   z++;
  }
 }

 else
 {
  for(int z=0; z<rank_ny; z++)
  {
   for(int j=m-1; j >= 0; j--)
   {
    b[z][j] = b[z][j] - b[z][k+rank_ystart0[i]]*pivotrow[j]/(*pivot);
   }
  }
 }
}

void solve_x(double** b,double* xsol,int rank_ny,int rank_ystart)
{
 for(int i=rank_ny-1; i>=0; i--)
 {
  double k=0.0;

  for(int j=0; j<n; j++)
  {
   k = k + b[i][j]*xsol[j];
  }

  xsol[i+rank_ystart] = (b[i][n]-k)/b[i][i+rank_ystart];
 }
}

int main(int argc,char* argv[])
{
 int rank,size;
 MPI_Init(&argc,&argv);
 MPI_Comm_size(MPI_COMM_WORLD,&size);
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);

 printf("rank: %d of %d\n",rank,size);

 int rank_ny,rank_ystart;
 
 if(rank < n % size)
 {
  rank_ny = n/size + 1;
  rank_ystart = rank * rank_ny;
 }
 
 else
 {
  rank_ny = n/size;
  rank_ystart = n - (size-rank)*rank_ny;
 }
 
 malloc2ddouble(&a, n, m);

 double *xsol = (double*) calloc(n,sizeof(double));

 pivotrow = (double*) malloc(m*sizeof(double));

 double *pivot = (double*) malloc(1*sizeof(double));
 
 FILE * fp;
 fp = fopen("matrix.txt", "r");

 for(int i = 0; i < n; i++)
 {
  for(int j = 0; j < m; j++)
  {
   fscanf(fp,"%lf ",&a[i][j]);
  }          
 }

 malloc2ddouble(&b, rank_ny, m);

 for(int z=0,k=rank_ystart; z<rank_ny && k<n; z++,k++)
 {
  for(int y=0; y<m; y++)
  {
   b[z][y] = a[k][y];
  }
 }

 int *rank_ny0 = (int*) malloc(size*sizeof(int));
 int *rank_ystart0 = (int*) malloc(size*sizeof(int));
 int tag;
 MPI_Status status;

 if(rank != 0)
 {
  tag = 1;
  MPI_Send(&rank_ny,1,MPI_INT,0,tag,MPI_COMM_WORLD);
  tag = 2;
  MPI_Send(&rank_ystart,1,MPI_INT,0,tag,MPI_COMM_WORLD);
 }

 if(rank==0)
 {
  tag = 1;
  for(int i=0; i<size; i++)
  {
   if(i==0)
   {
    rank_ny0[i] = rank_ny;
   }
   else
   { 
    MPI_Recv(&rank_ny0[i],1,MPI_INT,i,tag,MPI_COMM_WORLD,&status);
   }
  }

  tag = 2;
  for(int i=0; i<size; i++)
  {
   if(i==0)
   {
    rank_ystart0[i] = rank_ystart;
   }
   else
   { 
    MPI_Recv(&rank_ystart0[i],1,MPI_INT,i,tag,MPI_COMM_WORLD,&status);
   }
  }
 }

 MPI_Bcast(&rank_ny0[0],size,MPI_INT,0,MPI_COMM_WORLD);

 MPI_Bcast(&rank_ystart0[0],size,MPI_INT,0,MPI_COMM_WORLD);

 for(int i=0; i<size-1; i++) 
 {
  int k=0;

  while(k < rank_ny0[i])
  {
   if(i == rank)
   {
    broadcat(b,pivot,k,rank_ystart);
   }

   MPI_Bcast(pivot,1,MPI_DOUBLE,i,MPI_COMM_WORLD);

   MPI_Bcast(pivotrow,m,MPI_DOUBLE,i,MPI_COMM_WORLD);

   MPI_Barrier(MPI_COMM_WORLD);

   if(rank >= i)
   {
    gauss_ele(b,pivot,k,i,rank,rank_ny,rank_ystart0);
   }

   k++; 
  }
 }

 if(rank == size-1)
 {
  for(int i=0; i<rank_ny-1; i++)
  {
   for(int z= i+1; z<rank_ny; z++)
   {
    for(int j=m-1; j >= 0; j--)
    {
     b[z][j] = b[z][j] - b[z][i+rank_ystart]*b[i][j]/b[i][i+rank_ystart];
    }
   }
  }
 }

 for(int i=size-1; i>=0; i--)
 {
  if(rank == i)
  {
   solve_x(b,xsol,rank_ny,rank_ystart);
  }

  MPI_Bcast(xsol,n,MPI_DOUBLE,i,MPI_COMM_WORLD);
 }

 if(rank == size-1)
 {
  for(int i=0; i<n; i++)
  {
   printf("%g\n",xsol[i]);
  }
 }

 free2ddouble(&b); 
 free2ddouble(&a);
 free(xsol);
 free(pivot);
 free(pivotrow);

 MPI_Finalize();
 return 0;

}