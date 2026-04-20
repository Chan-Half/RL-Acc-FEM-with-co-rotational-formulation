"""
###########################################################################################
#  @copyright: Institute of Automation, Chinese Academy of Sciences
#  @filename:  fem_cuda_mesh.py
#  @brief:     fem cuda function of corotate mesh
#  @author:    Hao Chen
#  @version:   1.0
#  @date:      2025.12.19
#  @Email:     chen.hao2020@ia.ac.cn
#  @Note:      Remember to free up the memory allocated by `new`, otherwise a memory leak will occur.
###########################################################################################
"""
from pycuda.compiler import SourceModule
#use the cuda

def get_cuda_mesh():
    mod = SourceModule("""

#include <math.h>
typedef struct m32{
    int width;
    int height;
    int stride; 
    int __padding;    //为了和64位的elements指针对齐
    float* elements;
    __device__ m32(){};
    __device__ ~m32(){
        if(elements){
            delete[] elements;
            elements = NULL;
        }
    };
} Matrix;
typedef struct m64{
    int width;
    int height;
    int stride; 
    int __padding;    //为了和64位的elements指针对齐
    double* elements;
    __device__ m64(){};
    __device__ ~m64(){
        if(elements){
            delete[] elements;
            elements = NULL;
        }
    };
} Matrix64;

/**--------------------------------------------------------------------------------*/
// co-rotate frame
__device__ double vector_dot(double *a, double *b){
    //dot of two 3 dimension vector to get a double
    //int len_a = sizeof(a)/sizeof(a[0]);
    double c = 0;
    for(int i=0;i<3;i++){
        c += a[i]*b[i];
    }
    return c;
}
__device__ int outer(double *a, double *b, double *c){
    //outer multiply of two vector of 3 dimension
    c[0] = a[0]*b[0];
    c[1] = a[0]*b[1];
    c[2] = a[0]*b[2];
    c[3] = a[1]*b[0];
    c[4] = a[1]*b[1];
    c[5] = a[1]*b[2];
    c[6] = a[2]*b[0];
    c[7] = a[2]*b[1];
    c[8] = a[2]*b[2];
    return 0;
}

__device__ int multiply_3d(double *a, double *b, double *c)
{   
    double sum = 0;
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
        {   
            sum = 0;
            for(int k=0;k<3;k++)
            {
                sum += a[i*3+k]*b[k*3+j];
            }
            c[i*3+j] = sum;
        }
    return 0;
}

__device__ int theta_hat(double *theta, double *hat){
    
    // 反对称矩阵
    hat[0] = 0.0;
    hat[1] = -theta[2];
    hat[2] = theta[1];
    hat[3] = theta[2];
    hat[4] = 0.0;
    hat[5] = -theta[0];
    hat[6] = -theta[1];
    hat[7] = theta[0];
    hat[8] = 0.0;
    return 0;
}

__device__ int Ts(double *theta, Matrix* I, double *ts){
    // 旋转向量求导
    double theta_norm = sqrt(theta[0]*theta[0]+theta[1]*theta[1]+theta[2]*theta[2]);//np.linalg.norm(theta)
    if(theta_norm == 0){
        for(int i=0;i<9;i++){
            ts[i] = I->elements[i];
        }
        return 0;
    }
    double e[3];
    for(int i=0;i<3;i++){
    e[i] = theta[i] / theta_norm;
    }
    double td = sin(theta_norm / 2) / (theta_norm / 2);
    double te = sin(theta_norm) / theta_norm;
    double c[9] = {0.};
    outer(e,e,c);
    double d[9] = {0.};
    theta_hat(theta, d);
    
    for(int i=0;i<9;i++){
        ts[i] = te*I->elements[i] + (1-te)*c[i]+0.5*td*td*d[i];
    }
    return 0;
}

__device__ int Ts_inv(double *theta, Matrix* I, double *ts_inv){
    //旋转向量求导 inv
    double theta_norm = sqrt(theta[0]*theta[0]+theta[1]*theta[1]+theta[2]*theta[2]);//np.linalg.norm(theta)
    // assert theta_norm < 3.1415926535 * 2
    if(theta_norm == 0){
        for(int i=0;i<9;i++){
            ts_inv[i] = I->elements[i];
        }
        return 0;
    }
    // nambla = (2 * np.sin(theta_norm) - theta_norm * (1 + np.cos(theta_norm))) / (2 * theta_norm ** 2 * np.sin(theta_norm / 2))
    // hat = self.theta_hat(theta)
    double td = theta_norm / 2 / tan(theta_norm / 2);
    double the_the_out[9] = {0.};
    outer(theta, theta, the_the_out);
    double the_hat[9] = {0.};
    theta_hat(theta, the_hat);
    
    for(int i=0;i<9;i++){
        ts_inv[i] = td*I->elements[i] + (1-td)/pow(theta_norm,2)*the_the_out[i] - 0.5 * the_hat[i];
    }
    return 0;
    // return self.I + nambla*np.dot(hat, hat)- 0.5 * hat
}

__device__ void AxialAngle2Rotation(double* theta, double* result, Matrix *I){
        double theta_norm = sqrt(theta[0] * theta[0] + theta[1] * theta[1] + theta[2] * theta[2]);
        if(theta_norm == 0){
            for(int k=0;k<9;k++){
                result[k] = I->elements[k];
            }
        }
        else{
            double theta_hat1[9];
            theta_hat(theta, theta_hat1);
            double hat_hat_mul[9];
            multiply_3d(theta_hat1, theta_hat1, hat_hat_mul);
            double n1 = sin(theta_norm) / theta_norm;
            double n2 = pow((sin(theta_norm / 2) / theta_norm * 2), 2);
            for(int k=0;k<9;k++){
                result[k] = I->elements[k] + n1 * theta_hat1[k] + n2 * hat_hat_mul[k];
            }
        }
}

__global__ void update_Kh(Matrix64 *dl, Matrix64 *fl, Matrix64 *Kh, Matrix *I){
    int i = blockIdx.x; // self.node_number - 1
    int j = threadIdx.x; // 2
    double theta[3];
    double m[3];
    for(int k=0;k<3;k++){
        theta[k] = dl->elements[i*dl->width + 1 + j * 3 + k];
        m[k] = fl->elements[i*fl->width + 1 + j * 3 + k];
    }
    //if(i == 0 && j == 0){printf("theta: %.14f, %.14f, %.14f\\n", theta[0], theta[1], theta[2]); printf("m: %f, %f, %f\\n", m[0], m[1], m[2]);}
    double a = sqrt(theta[0]*theta[0]+theta[1]*theta[1]+theta[2]*theta[2]);//np.linalg.norm(theta)
    //if(i == 0 && j == 0)printf("a: %.14f\\n", a);
    // 零点求导不一定是单位矩阵
    if(a == 0)
        {
            for(int k=0;k<3;k++)
                for(int h=0;h<3;h++){
                    Kh->elements[i*Kh->height*Kh->width + (1 + j * 3 + k)*Kh->width + 1 + j * 3 + h] = I->elements[k*I->width+h];
                }
        }
    else{
        double nambla = (2. * sin(a) - a * (1. + cos(a))) / (2. * pow(a, 2.) * sin(a / 2.));
        // notation: there is a parameter of 8 in some paper like influence of symmetrisation, but not in corotation paper
        double miu = (a * (a + sin(a)) - 8.*sin(a / 2.)*sin(a / 2.)) / (4. * a*a*a*a * sin(a / 2.)*sin(a / 2.));
        if(miu > 10 || miu < -10)miu = 0.0;  //set minimal number to zero
        //if(i == 0 && j == 0){printf("nambla miu: %.14f, %.14f\\n", nambla, miu);}
        //-----------------------------------------------------------------------------------
        double k1[9]={0.};
        double theta_m_out[9];
        outer(theta, m, theta_m_out);
        double m_theta_out[9];
        outer(m, theta, m_theta_out);
        double theta_m_dot = vector_dot(theta, m);
        double theta_hat1[9];
        theta_hat(theta, theta_hat1);
        double m_hat[9];
        theta_hat(m, m_hat);
        double hat_hat_mul[9];
        multiply_3d(theta_hat1, theta_hat1, hat_hat_mul);
        double hat_hat_out_mul[9];
        multiply_3d(hat_hat_mul, m_theta_out, hat_hat_out_mul);
        for(int k=0;k<9;k++){
            k1[k] = nambla * (theta_m_out[k] - 2 * m_theta_out[k] + theta_m_dot * I->elements[k]) + miu * hat_hat_out_mul[k] - 0.5 * m_hat[k];
        }
        
        double kk[9];
        double ts_inv[9];
        Ts_inv(theta, I, ts_inv);
        multiply_3d(k1, ts_inv, kk);
        
        for(int k=0;k<3;k++)
            for(int h=0;h<3;h++){
                    Kh->elements[i*Kh->height*Kh->width + (1 + j * 3 + k)*Kh->width + 1 + j * 3 + h] = kk[k*3+h];
                }
        }
        
}
/**------------------------------------------------------------------------------------------------------------------*/

// 函数用于计算两个Matrix64矩阵的乘积
__device__ void matrix64Multiplication(Matrix64 *A, Matrix64 *B, Matrix64 *result){
    
    result->width = B->width;
    result->height = A->height;
    result->stride = B->stride;
    result->__padding = 0;
    result->elements = new double[result->height * result->width];

    for (int i = 0; i < A->height; ++i) {
        for (int j = 0; j < B->width; ++j) {
            double sum = 0.0;
            for (int k = 0; k < A->width; ++k) {
                sum += A->elements[i * A->width + k] * B->elements[k * B->width + j];
            }
            result->elements[i * result->width + j] = sum;
        }
    }

}
// 函数用于计算两个double矩阵的乘积
__device__ void doubleMatrixMultiplication(double *A,int A_height,int A_width, double *B, int B_height, int B_width, Matrix64 *result){

    result->width = B_width;
    result->height = A_height;
    result->stride = 0;
    result->__padding = 0;
    result->elements = new double[result->height * result->width];

    for (int i = 0; i < A_height; ++i) {
        for (int j = 0; j < B_width; ++j) {
            double sum = 0.0;
            for (int k = 0; k < A_width; ++k) {
                sum += A[i * A_width + k] * B[k * B_width + j];
            }
            result->elements[i * result->width + j] = sum;
        }
    }

}
// 函数用于计算Matrix64矩阵的转置
__device__ void matrixTranspose(Matrix64* A, Matrix64 *result){
    
    result->width = A->height; // 转置后的列数等于转置前的行数
    result->height = A->width; // 转置后的行数等于转置前的列数
    result->stride = A->height; // 假设转置后的矩阵也是行主序存储
    result->__padding = 0;
    result->elements = new double[result->height * result->stride];

    for (int i = 0; i < A->height; ++i) {
        for (int j = 0; j < A->width; ++j) {
            result->elements[j * result->width + i] = A->elements[i * A->stride + j];
        }
    }
}
// 函数用于计算double矩阵的转置
__device__ void doubleMatrixTranspose(double* A, int A_height, int A_width, Matrix64 *result){
    
    result->width = A_height; // 转置后的列数等于转置前的行数
    result->height = A_width; // 转置后的行数等于转置前的列数
    result->stride = A_height; // 假设转置后的矩阵也是行主序存储
    result->__padding = 0;
    result->elements = new double[result->height * result->width];

    for (int i = 0; i < A_height; ++i){
        for (int j = 0; j < A_width; ++j){
            result->elements[j * result->width + i] = A[i * A_width + j];
        }
    }
}

// vector cross
__device__ double* vector_cross(double *a, double *b){
    //dot of two 3 dimension vector to get a double
    //int len_a = sizeof(a)/sizeof(a[0]);
    double *c = new double[3];
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
    return c;
}

__global__ void update_R1R2(Matrix64* dx, Matrix64* R1, Matrix64* R2, Matrix *I){
    int i = blockIdx.x; // self.node_number - 1
    double theta1[3];
    double theta2[3];
    for(int j = 0; j < 3; j++){
        theta1[j] = dx->elements[i * 6 + 3 + j];
        theta2[j] = dx->elements[i * 6 + 9 + j];
    }
    double A2R1[9] = {0.0};
    double A2R2[9] = {0.0};
    AxialAngle2Rotation(theta1, A2R1, I);
    AxialAngle2Rotation(theta2, A2R2, I);
    
    double A2R1_R1_mul[9];
    double A2R2_R2_mul[9];
    double *R1_temp = &R1->elements[i*R1->height*R1->width];
    double *R2_temp = &R2->elements[i*R2->height*R2->width];
    
    multiply_3d(A2R1, R1_temp, A2R1_R1_mul);
    multiply_3d(A2R2, R2_temp, A2R2_R2_mul);
    
    for(int j = 0; j < 9; j++){
        R1->elements[i*R1->height*R1->width + j] = A2R1_R1_mul[j];
        R2->elements[i*R2->height*R2->width + j] = A2R2_R2_mul[j];
    }
    
}

__global__ void update_q(Matrix64* R1, Matrix64* R2, Matrix64* R0, Matrix64* q, Matrix64* q1, Matrix64* q2){
    int i = blockIdx.x;  //self.node_number - 1
    // 注意这里的乘法,经过测试暂未发现行列的bug
    double *q_p = &q->elements[i*q->width];
    double *q1_p = &q1->elements[i*q1->width];
    double *q2_p = &q2->elements[i*q2->width];
    double *R1_p = &R1->elements[i*R1->height*R1->width];
    double *R2_p = &R2->elements[i*R2->height*R2->width];
    double *R0_p = &R0->elements[i*R0->height*R0->width];
    for(int j = 0; j < 3; j++){
        q1_p[j] = 0.0;
        q2_p[j] = 0.0;
        for (int k = 0; k < 3; k++){
            q1_p[j] += R1_p[j*R1->width + k] * R0_p[k*R0->width + 1];
            q2_p[j] += R2_p[j*R2->width + k] * R0_p[k*R0->width + 1];
        }
        q_p[j] = 0.5*(q1_p[j] + q2_p[j]);
    }
}

__global__ void update_Rn(Matrix64* v1, Matrix64 *ln,  Matrix64 *ResultNode, Matrix64 *Rn, Matrix64 *q){
    // 注意Rn的更新方式可能会变
    int i = blockIdx.x;  //self.node_number - 1
    double *r1 = &ResultNode->elements[i*ResultNode->width];
    double *v_p = &v1->elements[i*v1->width];
    for(int j = 0; j < 3; j++){
        v_p[j] = (r1[j+3]-r1[j]) / ln->elements[i];
    }
    double *Rn_p = &Rn->elements[i*Rn->height*Rn->width];
    double *q_p = &q->elements[i*q->width];
    double *cross_2 = vector_cross(v_p, q_p);
    //double len_cross_2 = sqrt(pow(cross_2[0], 2) + pow(cross_2[1], 2) + pow(cross_2[2], 2));
    //for(int j = 0; j < 3; j++){
        // cross_2[j] /= len_cross_2;
    // }
    double *cross_1 = vector_cross(cross_2, v_p);
    for(int j = 0; j < 3; j++){
        Rn_p[j*Rn->width] = v_p[j];
        Rn_p[j*Rn->width + 1] = cross_1[j];
        Rn_p[j*Rn->width + 2] = cross_2[j];
    }
    delete[] cross_2;
    cross_2 = NULL;
    delete[] cross_1;
    cross_1 = NULL;
}

__global__ void update_Tg(Matrix64 *Tg, Matrix64 *Rn){
    // 更新于Rn后面
    int i = blockIdx.x;  //self.node_number - 1
    double *Tg_p = &Tg->elements[i*Tg->height*Tg->width];
    double *Rn_p = &Rn->elements[i*Rn->height*Rn->width];
    for(int j=0;j<4;j++){
        for(int k=0;k<3;k++){
            for(int m=0;m<3;m++){
                Tg_p[(j*3 + k)*Tg->width + j*3 + m] = Rn_p[k*Rn->width + m];
            }
        }
    }
}

__global__ void update_Gt(Matrix64 *Rn, Matrix64 *q, Matrix64 *q1, Matrix64 *q2, Matrix64 *nambla, Matrix64 *ln, Matrix64 *Gt){
    int i = blockIdx.x;  //self.node_number - 1
    // 这里要不要转置,以及公式中到底是０和１下标还是１和２
    double *Rn_p = &Rn->elements[i*Rn->height*Rn->width];
    double *q_p = &q->elements[i*q->width];
    double *q1_p = &q1->elements[i*q1->width];
    double *q2_p = &q2->elements[i*q2->width];
    double Rntq[3] = {0.0};
    double Rntq1[3] = {0.0};
    double Rntq2[3] = {0.0};
    for(int j=0;j<3;j++){
        for(int k=0;k<3;k++){
            Rntq[j] += Rn_p[k*Rn->width + j] * q_p[k];
            Rntq1[j] += Rn_p[k*Rn->width + j] * q1_p[k];
            Rntq2[j] += Rn_p[k*Rn->width + j] * q2_p[k];
        }
    }
    //if(i == 0){printf("q1_p: %.24f, %.24f, %.24f\\n", q1_p[0], q1_p[1], q1_p[2]);}
    nambla->elements[i] = Rntq[0] / Rntq[1];

    double *Gt_p = &Gt->elements[i*Gt->height*Gt->width];
    Gt_p[2] = Rntq[0] / Rntq[1] / ln->elements[i];
    Gt_p[3] = Rntq1[1] / Rntq[1] / 2.0;
    Gt_p[4] = -Rntq1[0] / Rntq[1] / 2.0;
    Gt_p[8] = -Rntq[0] / Rntq[1] / ln->elements[i];
    Gt_p[9] = Rntq2[1] / Rntq[1] / 2.0;
    Gt_p[10] = -Rntq2[0] / Rntq[1] / 2.0;

    Gt_p[1*Gt->width + 2] = 1.0 / ln->elements[i];
    Gt_p[1*Gt->width + 8] = -1.0 / ln->elements[i];
    Gt_p[2*Gt->width + 1] = -1.0 / ln->elements[i];
    Gt_p[2*Gt->width + 7] = 1.0 / ln->elements[i];
}

__global__ void update_P(Matrix64 *P, Matrix64 *Gt){
    int i = blockIdx.x;  //self.node_number - 1
    double *P_p = &P->elements[i*P->height*P->width];
    double *Gt_p = &Gt->elements[i*Gt->height*Gt->width];
    for(int j=0;j<6;j++){
        for(int k=0;k<12;k++){
            P_p[j*P->width+k] = -Gt_p[(j%3)*Gt->width + k];
            if(j<3){
                if(2<k && k<6 && j == (k%3)){P_p[j*P->width+k] += 1.0;}
            }
            else{
                if(8<k && j - 3 == (k%3)){P_p[j*P->width+k] += 1.0;}
            }
        }
    }
}

__global__ void update_B(Matrix64* Bm, Matrix64* Bm_local, Matrix64* v1, Matrix64* P, Matrix64* Tg, Matrix64* dl, Matrix *I){
    int i = blockIdx.x;
    //update Bm_local
    Bm_local->elements[i*Bm_local->height*Bm_local->width] = 1;
    double Ts_inv1[9];
    double Ts_inv2[9];
    double theta1[3];
    double theta2[3];
    for(int k=0;k<3;k++){
        theta1[k] = dl->elements[i*dl->width + 1 + k];
        theta2[k] = dl->elements[i*dl->width + 4 + k];
    }
    Ts_inv(theta1, I, Ts_inv1);
    Ts_inv(theta2, I, Ts_inv2);
    for(int j=0;j<3;j++){
        for(int k=0;k<3;k++){
            Bm_local->elements[i*Bm_local->height*Bm_local->width + (1 + j)*Bm_local->width + 1 + k] = Ts_inv1[j*3 + k];
            Bm_local->elements[i*Bm_local->height*Bm_local->width + (4 + j)*Bm_local->width + 4 + k] = Ts_inv2[j*3 + k];
        }
    }
    
    //update Bm
    for (int j = 0; j < 3; j++){
        Bm->elements[i*Bm->height*Bm->width + j] = -v1->elements[i*v1->width + j];
        Bm->elements[i*Bm->height*Bm->width + 6 + j] = v1->elements[i*v1->width + j];
    }
    
    Matrix64 *Tg_T = new Matrix64;
    Matrix64 *P_Tg_T_mul = new Matrix64;
    doubleMatrixTranspose(&Tg->elements[i*Tg->height*Tg->width], Tg->height, Tg->width, Tg_T);
    doubleMatrixMultiplication(&P->elements[i*P->height*P->width],P->height,P->width, Tg_T->elements, Tg_T->height, Tg_T->width, P_Tg_T_mul);

    for (int j = 0; j < 6; j++){
        for (int k = 0; k < 12; k++){
            Bm->elements[i * Bm->height*Bm->width + (j+1) *Bm->width + k] = P_Tg_T_mul->elements[j*P_Tg_T_mul->width + k];
        }
    }
    delete Tg_T;
    delete P_Tg_T_mul;
}
        """)

    return mod
