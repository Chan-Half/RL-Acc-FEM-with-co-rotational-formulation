"""
###########################################################################################
#  @copyright: 中国科学院自动化研究所智能微创医疗技术实验室
#  @filename:  fem_cuda_SourceModule.py
#  @brief:     fem cuda function
#  @author:    Hao Chen
#  @version:   1.0
#  @date:      2023.04.03
#  @Email:     chen.hao2020@ia.ac.cn
###########################################################################################
"""
from pycuda.compiler import SourceModule


def get_cuda_SourceModule():
    mod = SourceModule("""

#include <math.h>
typedef struct f32{
    int width;
    int height;
    int stride; 
    int __padding;    //为了和64位的elements指针对齐
    float* elements;
    __device__ f32(){};
    __device__ ~f32(){
        if(elements){
            delete[] elements;
            elements = NULL;
        }
    };
} Matrix;
typedef struct i32{
    int width;
    int height;
    int stride; 
    int __padding;    //为了和64位的elements指针对齐
    int* elements;
    __device__ i32(){};
    __device__ ~i32(){
        if(elements){
            delete[] elements;
            elements = NULL;
        }
    };
} Matrix_i32;
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
// 设置空matrix矩阵
__device__ Matrix SetEmptyMatrix(int height, int width, int stride) 
{
    Matrix A;
    A.width    = width;
    A.height   = height;
    A.stride   = stride;
    A.elements = new float[width*height];
    return A;
}
// 读取矩阵元素
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// 赋值矩阵元素
__device__ void SetElement(Matrix A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}

__device__ void matrix_multiply(Matrix *dest, Matrix *a, Matrix *b)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int width = a->width;

    float sum = 0;
    for(int k=0;k<width;k++)
    {
        sum += a->elements[i*width+k]*b->elements[k*width+j];
    }
    dest->elements[i*width+j] = sum;

}

__device__ void matrix_add(Matrix *dest, Matrix *a, Matrix *b)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    dest->elements[i*a->width+j] = a->elements[i*a->width+j] + a->elements[i*a->width+j];
}
__device__ volatile int g_mutex;

// GPU lock-based synchronization function
__device__ void __gpu_sync(int goalVal)
{
    // thread ID in a block
    int tid_in_block = threadIdx.x * blockDim.y + threadIdx.y;
    // only thread 0 is used for synchronization
    if (tid_in_block == 0)
    {
        atomicAdd((int*) &g_mutex, 1);

        // only when all blocks add 1 go g_mutex
        // will g_mutex equal to goalVal
        while (g_mutex != goalVal)
        {
            // Do nothing here
        }
    }
    __syncthreads();
}

__device__ bool AreaMth(float *xtri, float *pNode, float distance1)
{   
    // Compute vectors

    float v[9];
    for(int m=0;m<3;m++){
        for(int n=0;n<3;n++){
            v[m*3+n]=xtri[m*3+n]-pNode[n];
        }
    }
    //if(threadIdx.x==25&&blockIdx.x==25){printf("v is %f,%f,%f\\n", v[0], v[1], v[2]);}
    //if(threadIdx.x==25&&blockIdx.x==25){printf("v is %f,%f,%f\\n", v[3], v[4], v[5]);}
    //if(threadIdx.x==25&&blockIdx.x==25){printf("v is %f,%f,%f\\n", v[6], v[7], v[8]);}
    float vn1 = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);if(vn1==0)vn1 = 0.000000001;
    float vn2 = sqrt(v[3]*v[3]+v[4]*v[4]+v[5]*v[5]);if(vn2==0)vn2 = 0.000000001;
    float vn3 = sqrt(v[6]*v[6]+v[7]*v[7]+v[8]*v[8]);if(vn3==0)vn3 = 0.000000001;
    float theta11 = acos((v[0]*v[3]+v[1]*v[4]+v[2]*v[5])/vn1/vn2);
    float theta12 = acos((v[6]*v[3]+v[7]*v[4]+v[8]*v[5])/vn3/vn2);
    float theta13 = acos((v[0]*v[6]+v[1]*v[7]+v[2]*v[8])/vn1/vn3);

    //if(threadIdx.x==25&&blockIdx.x==250){printf("dis1 is %f,%f,%f\\n", theta11, theta12, theta13);}
    if(distance1 < 0.0){
        if((2 * 3.1415926535 - (theta11 + theta12 + theta13)) < 0.05){
        //printf("theta11 %f, theta12 %f, theta13 %f\\n",theta11, theta12, theta13);
        return 1;}
    else{
        return 0;};
    }
    else{
    if((2 * 3.1415926535 - (theta11 + theta12 + theta13)) < 3.5){
        //printf("theta11 %f, theta12 %f, theta13 %f\\n",theta11, theta12, theta13);
        return 1;}
    else{
        return 0;};
    }
}

__device__ float pPlane(float *Node, float *Normal, float dis0, float *pNode)
{
    // 求投影点及点到面的距离

    float distance1 = Node[0]*Normal[0] + Node[1]*Normal[1] + Node[2]* Normal[2] - dis0;
    for(int i=0;i<3;i++){
        pNode[i] = Node[i] - distance1*Normal[i];
    }
    distance1 = fabs(distance1);
    return distance1;
}

__global__ void SolveModel(Matrix *ResultNode, Matrix *x, Matrix *xtriangle, Matrix *gNode,
                            Matrix *vNormal, Matrix *distance0,
                            Matrix *constraint_triangle, int* Ax, Matrix *A, Matrix *b, Matrix* AdjacentTriangle, int* Adjx, float *delta_h_time)
{

    //clock_t starttime = clock();
    int i = blockIdx.x; // bool_nearpath->height||xtriangle->height
    int j = threadIdx.x; // gNode->height

    // 求解约束模型
    __shared__ float *Normal;
    __shared__ float dis0;
    __shared__ float h;
    if(j==blockDim.x-1){
        Normal = &vNormal->elements[i*vNormal->width];
        dis0 = distance0->elements[i];
        h = delta_h_time[0];
    }
    __shared__ float xtri[9];
    if(j<9){
        xtri[j] = x->elements[int(xtriangle->elements[i*xtriangle->width+j/3])*x->width+j%3];

    }

    __syncthreads();

    float Node[3];
    for(int k=0;k<3;k++){
        Node[k]=ResultNode->elements[j*ResultNode->width+k];
    }
    float pNode[3];
    float distance1 = pPlane(Node, Normal, dis0, pNode);
    if(distance1< 2.5 && AreaMth(xtri, pNode, distance1)){
        int m = atomicAdd(Ax, 1);
        int t = A->width*m;
        int n = A->width*m + j*6;
        for(int k = 0;k<A->width;k++){
            A->elements[t+k]=0;
        }
        for(int k=0;k<3;k++){
            A->elements[n+k] = h * Normal[k];
        }
        b->elements[m] = (dis0 - Normal[0]*gNode->elements[j*gNode->width]-Normal[1]*gNode->elements[j*gNode->width+1]-Normal[2]*gNode->elements[j*gNode->width+2]); //注意这里加了范围0.2
        constraint_triangle->elements[m] = i;
    }
}

__device__ double vector_dot(double *a, double *b){
    //dot of two 3 dimension vector to get a double
    //int len_a = sizeof(a)/sizeof(a[0]);
    double c = 0;
    for(int i=0;i<3;i++){
        c += a[i]*b[i];
    }
    return c;
}

__device__ bool is_in_face(double *xtri, double *crossNode)
{
    
    double AP[3], AB[3], AC[3];
    for(int n=0;n<3;n++){
        AP[n] = crossNode[n] - xtri[n];
        AB[n] = xtri[n+3] - xtri[n];
        AC[n] = xtri[n+6] - xtri[n];
    }
    
    float f_i=vector_dot(AP,AC)*vector_dot(AB,AB)-vector_dot(AP,AB)*vector_dot(AC,AB);
    float f_j=vector_dot(AP,AB)*vector_dot(AC,AC)-vector_dot(AP,AC)*vector_dot(AB,AC);
    float f_d=vector_dot(AC,AC)*vector_dot(AB,AB)-vector_dot(AC,AB)*vector_dot(AC,AB);
    
    if(f_i>=0 && f_j>=0 && f_i+f_j-f_d<=0){
        return 1;
    }
    else{
        return 0;
    }
    
    /**
    double v[9];
    for(int m=0;m<3;m++){
        for(int n=0;n<3;n++){
            v[m*3+n]=xtri[m*3+n]-crossNode[n];
        }
    }
    float vn1 = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);if(vn1==0)vn1 = 0.000000001;
    float vn2 = sqrt(v[3]*v[3]+v[4]*v[4]+v[5]*v[5]);if(vn2==0)vn2 = 0.000000001;
    float vn3 = sqrt(v[6]*v[6]+v[7]*v[7]+v[8]*v[8]);if(vn3==0)vn3 = 0.000000001;
    float theta11 = acos((v[0]*v[3]+v[1]*v[4]+v[2]*v[5])/vn1/vn2);
    float theta12 = acos((v[6]*v[3]+v[7]*v[4]+v[8]*v[5])/vn3/vn2);
    float theta13 = acos((v[0]*v[6]+v[1]*v[7]+v[2]*v[8])/vn1/vn3);
    if((2 * 3.1415926535 - (theta11 + theta12 + theta13)) < 4.5){
        return true;}
    else{
        return false;};
    */
}
__device__ double get_dt(double *Node, double *v, double *a, double *Normal, double dis0, double *crossNode, double h)
{
    double aa = 0.5*(a[0]*Normal[0] + a[1]*Normal[1] + a[2]* Normal[2]);
    double bb = v[0]*Normal[0] + v[1]*Normal[1] + v[2]* Normal[2];
    double cc = (Node[0]*Normal[0] + Node[1]*Normal[1] + Node[2]* Normal[2]) - dis0;
    double dt;
    if(aa == 0){
        if(bb==0){
            dt = 1000000;
        }
        else{
            dt = - cc/bb;
        }
    }
    else if(aa > 0){
        double m = bb*bb - 4 * aa * cc;
        if(cc > 0){
            if(m<0){dt = 1000000;}
            else{
                double dt1 = (-bb - sqrt(m))/(2*aa);
                double dt2 = (-bb + sqrt(m))/(2*aa);
                dt = min(dt1, dt2);
            }
        }
        else{
            double dt1 = (-bb - sqrt(m))/(2*aa);
            double dt2 = (-bb + sqrt(m))/(2*aa);
            dt = max(dt1, dt2);
        }
    }
    else{
        double m = bb*bb - 4 * aa * cc;
        if(cc < 0){
            if(m<0){dt = 1000000;}
            else{
                double dt1 = (-bb - sqrt(m))/(2*aa);
                double dt2 = (-bb + sqrt(m))/(2*aa);
                dt = min(dt1, dt2);
            }
        }
        else{
            double dt1 = (-bb - sqrt(m))/(2*aa);
            double dt2 = (-bb + sqrt(m))/(2*aa);
            dt = max(dt1, dt2);
        }
    }
    if(dt <= 2*h){
        for(int i=0;i<3;i++){
            crossNode[i] = Node[i] + dt*v[i] + 0.5*dt*dt* a[i];
        }
    }
    else{
        for(int i=0;i<3;i++){
            crossNode[i] = 1000000;
        }
    }
    
    return dt;
}

__global__ void getConstraint(int* Ax, Matrix64 *ResultNode, Matrix64 *V, Matrix64 *M_A, Matrix64 *X, Matrix_i32 *Face, 
                            Matrix64 *vNormal, Matrix64 *distance0,
                            Matrix64 *constraint_triangle, Matrix64 *A, Matrix64 *b, int* Adjx, Matrix64 *delta_h_time)
{
    int i = blockIdx.x; // xtriangle->height
    int j = threadIdx.x; // gNode->height
    //if(i==3&&j==99)printf("%d,%d,%d\\n", Face->elements[i*Face->width+0],Face->elements[i*Face->width+1],Face->elements[i*Face->width+2]);
    // 求解约束模型
    __shared__ double *Normal;
    __shared__ double dis0;
    __shared__ double h;
    __shared__ double beta;
    __shared__ double xtri[9];
    if(j==blockDim.x-1){
        Normal = &vNormal->elements[i*vNormal->width];
        dis0 = distance0->elements[i];
        h = delta_h_time->elements[0];
        beta = 0.25 * pow(1.0 + 0.3, 2);
        for(int k=0;k<3;k++)
            for(int m=0;m<3;m++){
                xtri[k*3+m] = X->elements[int(Face->elements[i*Face->width+k])*X->width+m];
            }
    }
    //if(j<9){
    //    xtri[j] = X->elements[int(Face->elements[i*Face->width+j/3])*X->width+j%3];
    //}
    __syncthreads();
    double Node[3];
    double v[3];
    double a[3];
    for(int k=0;k<3;k++){
        Node[k]=ResultNode->elements[j*ResultNode->width+k];
        v[k]=V->elements[j*6+k];
        a[k]=M_A->elements[j*6+k];
    }
    double crossNode[3];
    double dt = get_dt(Node, v, a, Normal, dis0, crossNode, h);
    //if(i==3&&j==99){printf("%f,%f,%f,%f,%f,%f,%f,%f,%f\\n", xtri[0],xtri[1],xtri[2],xtri[3],xtri[4],xtri[5],xtri[6],xtri[7],xtri[8]);}
    //if(i==0&&j==99)printf("%f\\n", dt);
    if(0 <= dt && dt <= 2*h && is_in_face(xtri, crossNode)){
        int m = atomicAdd(Ax, 1);
        int t = A->width*m;
        int n = A->width*m + j*6;
        
        for(int k = 0;k<A->width;k++){
            A->elements[t+k]=0;
        }
        for(int k=0;k<3;k++){
            A->elements[n+k] = h *h * Normal[k] * beta;
        }
        b->elements[m] = dis0 - Normal[0]*(Node[0]+v[0]*h + (0.5-beta)*a[0]*h*h)-Normal[1]*(Node[1]+v[1]*h + (0.5-beta)*a[1]*h*h)-Normal[2]*(Node[2]+v[2]*h + (0.5-beta)*a[2]*h*h);
        constraint_triangle->elements[m] = i;
    }
}

__device__ double get_dt_x(double *Node, double *v, double *a, double *Normal, double dis0, double *crossNode, double h)
{
    double bb = v[0]*Normal[0] + v[1]*Normal[1] + v[2]* Normal[2];
    double cc = dis0 -  (Node[0]*Normal[0] + Node[1]*Normal[1] + Node[2]* Normal[2]);
    double dt;
    if(bb == 0){
        dt = 10000000;
    }
    else{
        dt = cc / bb;
    }
    if(dt <= 2*h && dt >= -2*h){
        for(int i=0;i<3;i++){
            crossNode[i] = Node[i] + dt*v[i];
        }
    }
    else{
        for(int i=0;i<3;i++){
            crossNode[i] = 1000000;
        }
    }
    return dt;
}

__global__ void getConstraint_x(int* Ax, Matrix64 *ResultNode, Matrix64 *V, Matrix64 *M_A, Matrix64 *X, Matrix_i32 *Face, 
                            Matrix64 *vNormal, Matrix64 *distance0,
                            Matrix64 *constraint_triangle, Matrix64 *A, Matrix64 *b, int* Adjx, Matrix64 *delta_h_time)
{
    int i = blockIdx.x; // xtriangle->height
    int j = threadIdx.x; // gNode->height
    //if(i==3&&j==99)printf("%d,%d,%d\\n", Face->elements[i*Face->width+0],Face->elements[i*Face->width+1],Face->elements[i*Face->width+2]);
    // 求解约束模型
    __shared__ double *Normal;
    __shared__ double dis0;
    __shared__ double h;
    __shared__ double beta;
    __shared__ double xtri[9];
    if(j==blockDim.x-1){
        Normal = &vNormal->elements[i*vNormal->width];
        dis0 = distance0->elements[i];
        h = delta_h_time->elements[0];
        beta = 0.25 * pow(1.0 + 0.3, 2);
        for(int k=0;k<3;k++)
            for(int m=0;m<3;m++){
                xtri[k*3+m] = X->elements[int(Face->elements[i*Face->width+k])*X->width+m];
            }
    }
    //if(j<9){
    //    xtri[j] = X->elements[int(Face->elements[i*Face->width+j/3])*X->width+j%3];
    //}
    __syncthreads();
    double Node[3];
    double v[3];
    double a[3];
    for(int k=0;k<3;k++){
        Node[k]=ResultNode->elements[j*ResultNode->width+k];
        v[k]=V->elements[j*6+k];
        a[k]=M_A->elements[j*6+k];
    }
    double crossNode[3];
    double dt = get_dt_x(Node, v, a, Normal, dis0, crossNode, h);
    //if(i==3&&j==99){printf("%f,%f,%f,%f,%f,%f,%f,%f,%f\\n", xtri[0],xtri[1],xtri[2],xtri[3],xtri[4],xtri[5],xtri[6],xtri[7],xtri[8]);}
    //if(i==0&&j==99)printf("%f\\n", dt);
    if(-2*h <= dt && dt <= 2*h && is_in_face(xtri, crossNode)){
        int m = atomicAdd(Ax, 1);
        int t = A->width*m;
        int n = A->width*m + j*6;
        
        for(int k = 0;k<A->width;k++){
            A->elements[t+k]=0;
        }
        for(int k=0;k<3;k++){
            A->elements[n+k] = Normal[k];
        }
        b->elements[m] = dis0 - Normal[0]*Node[0]-Normal[1]*Node[1]-Normal[2]*Node[2];
        constraint_triangle->elements[m] = i;
    }
}

__global__ void SearchAdjacentTriangle(Matrix *xtriangle, Matrix* AdjacentTriangle, int* Adjx)
{
    int i = blockIdx.x; // xtriangle->height
    int j = threadIdx.x; // 1024
    __shared__ float *xtri;
    if(j==0){
        xtri = &xtriangle->elements[i*xtriangle->width];
    }

    __syncthreads();
    float kdim = gridDim.x/blockDim.x+1;
    for(int k =0;k<kdim;k++)
    {
        int m = k*blockDim.x+j;
        if(m < xtriangle->height&&m!=i)
        {
            float* xtri2 = &xtriangle->elements[m*xtriangle->width];
            bool a = true;
            for(int p = 0;p<3;p++)
            {
                for(int q = 0;q<3;q++)
                {
                    if(xtri[p]==xtri2[q]&&a)
                    {
                        int ax = atomicAdd(&Adjx[i], 1);
                        AdjacentTriangle->elements[i*AdjacentTriangle->width+ax] = m;
                        // 这里是错的，因为会加入本身三次，三个顶点分别和自己相等
                        a = false;
                    }
                }
            }
        }
    }

}

__global__ void TransformMatrix(Matrix64 *RNode, Matrix64 *T)
{
    /**
    #  计算单元的坐标转换矩阵( 局部坐标 -> 整体坐标 )
    #  输入参数
    #      ie  ----- 节点号
    #  返回值
    #      T ------- 从局部坐标到整体坐标的坐标转换矩阵
    # global gElement, gNode

    xi = gNode[gElement[ie][0] - 1][0]
    yi = gNode[gElement[ie][0] - 1][1]
    zi = gNode[gElement[ie][0] - 1][2]
    xj = gNode[gElement[ie][1] - 1][0]
    yj = gNode[gElement[ie][1] - 1][1]
    zj = gNode[gElement[ie][1] - 1][2]
    L = ((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2) ** (1 / 2)
    lx = (xj - xi) / L
    mx = (yj - yi) / L
    nx = (zj - zi) / L
    */

    int i = blockIdx.x; // node_number-1
    int j = threadIdx.x; // 12
    int k = threadIdx.y; // 12

    __shared__ double L, lx, mx, nx;
    if(j==0){
        lx = (RNode->elements[(i + 1)*RNode->width] - RNode->elements[(i)*RNode->width]);
        mx = (RNode->elements[(i + 1)*RNode->width+1] - RNode->elements[(i)*RNode->width+1]);
        nx = (RNode->elements[(i + 1)*RNode->width+2] - RNode->elements[(i)*RNode->width+2]);
        L = sqrt(lx *lx + mx *mx + nx*nx);
        lx = lx/L;
        mx = mx/L;
        nx = nx/L;
    }
    __syncthreads();
    
    if(j%3==0 && j==k)T->elements[i*T->height*T->width+j*T->width+k] = lx;
    if(j%3==0 && k==j+1)T->elements[i*T->height*T->width+j*T->width+k] = -nx * lx / sqrt(lx*lx + mx*mx);
    if(j%3==0 && k==j+2)T->elements[i*T->height*T->width+j*T->width+k] = mx / sqrt(lx*lx + mx*mx);
    if(j%3==1 && k==j-1)T->elements[i*T->height*T->width+j*T->width+k] = mx;
    if(j%3==1 && j==k)T->elements[i*T->height*T->width+j*T->width+k] = -nx * mx / sqrt(lx*lx + mx*mx);
    if(j%3==1 && k==j+1)T->elements[i*T->height*T->width+j*T->width+k] = -lx / sqrt(lx*lx + mx*mx);
    if(j%3==2 && k==j-2)T->elements[i*T->height*T->width+j*T->width+k] = nx;
    if(j%3==2 && k==j-1)T->elements[i*T->height*T->width+j*T->width+k] = sqrt(lx*lx + mx*mx);
    if(j%3==2 && j==k)T->elements[i*T->height*T->width+j*T->width+k] = 0.0;

    /**
    T[i] = np.array(
        [[lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2), mx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0, 0,
        0, 0, 0],
        [mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2), -lx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0,
        0, 0, 0, 0],
        [nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2), mx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0,
        0, 0, 0],
        [0, 0, 0, mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2), -lx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0,
        0, 0, 0, 0],
        [0, 0, 0, nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2), mx / (lx ** 2 + mx ** 2) ** (1 / 2),
        0, 0, 0],
        [0, 0, 0, 0, 0, 0, mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2),
        -lx / (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0],
        [0, 0, 0, 0, 0, 0, nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, lx, -nx * lx / (lx ** 2 + mx ** 2) ** (1 / 2),
        mx / (lx ** 2 + mx ** 2) ** (1 / 2)],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, mx, -nx * mx / (lx ** 2 + mx ** 2) ** (1 / 2),
        -lx / (lx ** 2 + mx ** 2) ** (1 / 2)],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, nx, (lx ** 2 + mx ** 2) ** (1 / 2), 0]])
    */

}


        """)

    return mod
