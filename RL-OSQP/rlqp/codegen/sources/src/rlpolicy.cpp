#include "rlpolicy.h"
#include <torch/script.h>
#include <Eigen/Dense>
#include <chrono>
#include <ATen/cuda/CUDAContext.h>

using Module = torch::jit::script::Module;

namespace rlqp {
    struct Policy {
        Module module_;
        at::Tensor gpu_inputs; // 预分配的GPU输入张量
        at::Tensor cpu_buffer; // 预分配的CPU缓冲区
        int current_m = -1;    // 当前使用的m值
    };

    struct Policy_cpu {
        Module baseModule_;
        Module module_;
    };
}

void *rl_policy_load(const char *module_path) {
    if (!module_path || !module_path[0]) return nullptr;

    rlqp::Policy* policy = new rlqp::Policy();
    try {
        // 加载模型到GPU
        policy->module_ = torch::jit::load(
            module_path, 
            torch::Device(torch::kCUDA)
        );
        policy->module_.eval();
        
        // 初始时不为缓冲区分配内存
        policy->current_m = -1;
        // c_print("Model loaded to GPU: %s \n", module_path);

        // 模型加载后进行预热
        constexpr int dummy_m = 1024; // 假设最大支持的m值
        std::vector<torch::jit::IValue> dummy_inputs{
            torch::zeros({dummy_m, 6}, torch::dtype(torch::kFloat32).device(torch::kCUDA))
        };
        for (int i = 0; i < 5; ++i) { // 多次调用确保完整初始化
            policy->module_.forward(dummy_inputs);
        }
        c10::cuda::device_synchronize(); // 确保预热完成
        // c_print("Model pre-warmed successfully.\n");

        return policy;
    } catch (const c10::Error& ex) {
        std::clog << "Load error: " << ex.what() << std::endl;
        delete policy;
        return nullptr;
    }
}

int rl_policy_compute_vec(OSQPWorkspace* work) {
    using namespace Eigen;
    using QPVec = Array<c_float, Dynamic, 1>;
    using RLVec = Array<float, Dynamic, 1>;
    // auto t1 = std::chrono::high_resolution_clock::now();
    
    rlqp::Policy *policy = static_cast<rlqp::Policy*>(work->rl_rho_policy);
    if (!policy) return 1;

    const int m = work->data->m;
    constexpr int stride = 6;
    constexpr int max_m = 10240; // 假设最大m值

    // 1. 检查是否需要重新分配内存，避免频繁分配
    if (policy->current_m == -1) {
        policy->gpu_inputs = torch::empty(
            {max_m, stride}, 
            torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false)
        );
        policy->cpu_buffer = torch::empty(
            {max_m}, 
            torch::dtype(torch::kFloat32).device(torch::kCPU)
        );
        policy->current_m = max_m;
        // c_print("Allocated buffers for max_m=%d\n", max_m);
    }

    // 2. 数据拷贝（非阻塞）
    auto copy_to_slice = [&](int col, const c_float* src) {
        float* buffer_ptr = policy->cpu_buffer.data_ptr<float>();
        for (int i = 0; i < m; ++i) {
            buffer_ptr[i] = static_cast<float>(src[i]);
        }
        
        // 同步拷贝到GPU
        policy->gpu_inputs.slice(1, col, col+1)
            .slice(0, 0, m) // 只使用当前的m值范围
            .copy_(policy->cpu_buffer.slice(0, 0, m).view({m, 1}), /*non_blocking=*/true);
    };

    copy_to_slice(0, work->Ax);
    copy_to_slice(1, work->y);
    copy_to_slice(2, work->z);
    copy_to_slice(3, work->data->l);
    copy_to_slice(4, work->data->u);
    copy_to_slice(5, work->rho_vec);
    
    std::vector<torch::jit::IValue> inputs{policy->gpu_inputs.slice(0, 0, m)};
    // auto t2 = std::chrono::high_resolution_clock::now();

    // 3. 前向传播
    at::Tensor piOutput = policy->module_.forward(inputs).toTensor();
    c10::cuda::device_synchronize(); // 同步以确保推理完成
    // auto t3 = std::chrono::high_resolution_clock::now();
    
    // 4. 处理输出
    piOutput = piOutput.slice(0, 0, m).to(torch::kCPU).contiguous(); // 只处理当前m值
    Eigen::Map<RLVec> output(piOutput.data_ptr<float>(), m);
    
    Map<QPVec>(work->rho_vec, m) = output.cast<c_float>();
    Map<QPVec>(work->rho_inv_vec, m) = 1 / output.cast<c_float>();
    
    // auto t4 = std::chrono::high_resolution_clock::now();
    
    // 5. 调试输出
    auto duration = [&](auto t_start, auto t_end) {
        return std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    };
    // c_print("Total time: %.3f ms, Copy: %.3f ms, Forward: %.3f ms, Output: %.3f ms\n",
    //     duration(t1, t4), duration(t1, t2), duration(t2, t3), duration(t3, t4));

    return 0;
}


int rl_policy_unload(void *ptr) {
    rlqp::Policy *policy = static_cast<rlqp::Policy*>(ptr);
    if (policy) {
        delete policy;
    }
    return 0;
}






/** ----------------------------------------------------sigma----------------------------------------------------------------- */ 

void *rl_policy_load_sigma(const char *module_path) {
    if (!module_path || !module_path[0]) return nullptr;

    rlqp::Policy* policy = new rlqp::Policy();
    try {
        // 加载模型到GPU
        policy->module_ = torch::jit::load(
            module_path, 
            torch::Device(torch::kCUDA)
        );
        policy->module_.eval();
        
        // 初始时不为缓冲区分配内存
        policy->current_m = -1;
        // c_print("Model loaded to GPU: %s \n", module_path);

        // 模型加载后进行预热
        constexpr int dummy_m = 1024; // 假设最大支持的m值
        std::vector<torch::jit::IValue> dummy_inputs{
            torch::zeros({dummy_m, 5}, torch::dtype(torch::kFloat32).device(torch::kCUDA))
        };
        for (int i = 0; i < 5; ++i) { // 多次调用确保完整初始化
            policy->module_.forward(dummy_inputs);
        }
        c10::cuda::device_synchronize(); // 确保预热完成
        // c_print("Model pre-warmed successfully.\n");

        return policy;
    } catch (const c10::Error& ex) {
        std::clog << "Load error: " << ex.what() << std::endl;
        delete policy;
        return nullptr;
    }
}

int rl_policy_compute_vec_sigma(OSQPWorkspace* work) {
    using namespace Eigen;
    using QPVec = Array<c_float, Dynamic, 1>;
    using RLVec = Array<float, Dynamic, 1>;
    // auto t1 = std::chrono::high_resolution_clock::now();

    //下面的policy也需要改
    
    rlqp::Policy *policy = static_cast<rlqp::Policy*>(work->rl_sigma_policy);
    if (!policy) return 1;

    const int m = work->data->n;
    constexpr int stride = 5;
    constexpr int max_m = 10240; // 假设最大m值

    // 1. 检查是否需要重新分配内存，避免频繁分配
    if (policy->current_m == -1) {
        policy->gpu_inputs = torch::empty(
            {max_m, stride}, 
            torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false)
        );
        policy->cpu_buffer = torch::empty(
            {max_m}, 
            torch::dtype(torch::kFloat32).device(torch::kCPU)
        );
        policy->current_m = max_m;
        // c_print("Allocated buffers for max_m=%d\n", max_m);
    }

    // 2. 数据拷贝（非阻塞）
    auto copy_to_slice = [&](int col, const c_float* src) {
        float* buffer_ptr = policy->cpu_buffer.data_ptr<float>();
        for (int i = 0; i < m; ++i) {
            buffer_ptr[i] = static_cast<float>(src[i]);
        }
        
        // 同步拷贝到GPU
        policy->gpu_inputs.slice(1, col, col+1)
            .slice(0, 0, m) // 只使用当前的m值范围
            .copy_(policy->cpu_buffer.slice(0, 0, m).view({m, 1}), /*non_blocking=*/true);
    };

    copy_to_slice(0, work->Px);
    copy_to_slice(1, work->Aty);
    copy_to_slice(2, work->data->q);
    copy_to_slice(3, work->P_diag);
    copy_to_slice(4, work->sigma_vec);
    
    std::vector<torch::jit::IValue> inputs{policy->gpu_inputs.slice(0, 0, m)};
    // auto t2 = std::chrono::high_resolution_clock::now();

    // 3. 前向传播
    at::Tensor piOutput = policy->module_.forward(inputs).toTensor();
    c10::cuda::device_synchronize(); // 同步以确保推理完成
    // auto t3 = std::chrono::high_resolution_clock::now();
    
    // 4. 处理输出
    piOutput = piOutput.slice(0, 0, m).to(torch::kCPU).contiguous(); // 只处理当前m值
    Eigen::Map<RLVec> output(piOutput.data_ptr<float>(), m);
    
    Map<QPVec>(work->sigma_vec, m) = output.cast<c_float>();
    // Map<QPVec>(work->rho_inv_vec, m) = 1 / output.cast<c_float>();
    
    // auto t4 = std::chrono::high_resolution_clock::now();
    
    // 5. 调试输出
    auto duration = [&](auto t_start, auto t_end) {
        return std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;
    };
    // c_print("Total time: %.3f ms, Copy: %.3f ms, Forward: %.3f ms, Output: %.3f ms\n",
    //          duration(t1, t4), duration(t1, t2), duration(t2, t3), duration(t3, t4));

    return 0;
}

