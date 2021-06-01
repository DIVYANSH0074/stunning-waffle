#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <sstream>
#include <fstream>
using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

int NX = 1000;
int NY = 1000;
int n = NX*NY; 

const int tx[] = {1, -1, 0, 0};
const int ty[] = {0, 0, 1, -1};

inline size_t scalar_index(int x,int y)
{
 return NX*y + x;
}

void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b)
{
 for(int y=0; y<NY; y++)
 {
  for(int x=0; x<NX; x++)
  {
   for(int d=0; d<4; d++)  
   { 
    int xm = x+tx[d];
    int ym = y+ty[d];

    if(xm >= 0 && ym >= 0 && xm <= NX-1 && ym <= NY-1)
    {
     coefficients.push_back(T(scalar_index(xm,ym),scalar_index(x,y),-1));
    } 
   }
   coefficients.push_back(T(scalar_index(x,y),scalar_index(x,y),10));
  }
 }

 for(int i=0; i<n; i++)
 {
  if(i%100 == 0)
  {
   b(i)=5000;
  }

  else
  {
   b(i)=100;
  }
 }
}

int main()
{
 std::vector<T> coefficients;            // list of non-zeros coefficients
 int nnz = 5*n;
 coefficients.reserve(nnz);
 Eigen::VectorXd b(n);

 clock_t start, end;
 double time;

 start = clock();

 buildProblem(coefficients, b);

 SpMat SparseA(n,n);
 SparseA.setFromTriplets(coefficients.begin(), coefficients.end());

 SparseA.makeCompressed();

 // cout << "Here is the Sparse matrix SparseA:\n"<< SparseA << endl;

 Eigen::BiCGSTAB<SparseMatrix<double> >  BCGST;

 BCGST.analyzePattern(SparseA);
 BCGST.compute(SparseA);

 if(BCGST.info()!=Eigen::Success) {
    std::cout << "Oh: Very bad" <<"\n";
 }
 else{
     std::cout<<"okay computed"<<"\n";
 }

 Eigen::VectorXd X;
 X = BCGST.solve(b);

 // cout << "Here is the Vector x using BICGSTAB :\n"<< X << endl;

 end = clock();

 time = ((double) (end-start)) / CLOCKS_PER_SEC;
 printf("%f\n",time);

 std::ofstream out_mat;
 out_mat.open("Mat.txt", std::ios::trunc);
 out_mat<< X << std::endl;
 out_mat.close();

 return 0;
}
