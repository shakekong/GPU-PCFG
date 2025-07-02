#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
using namespace std;


// 添加阈值常量
#define MIN_PARALLEL_SIZE 100 // 小于此数量的PT用CPU处理


__global__ void generate_kernel_single(
    char* d_values, int* d_lengths, int* d_offsets, 
    char* d_output, int max_pw_len, int N
);
__global__ void generate_kernel_multi(
    const char* d_guess, int guess_len,
    char* d_values, int* d_lengths, int* d_offsets,
    char* d_output, int max_pw_len, int N
);




void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


void PriorityQueue::Generate(PT pt) {
    CalProb(pt);
    const int MAX_PW_LEN = 256; // 最大口令长度
    segment* a = nullptr;
    int N = 0;
    string fixed_part;

    // 确定segment类型和长度
    if (pt.content.size() == 1) {
        if (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        else if (pt.content[0].type == 3) a = &m.symbols[m.FindSymbol(pt.content[0])];
        N = pt.max_indices[0];
    } else {
        // 构建固定部分
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (seg_idx == pt.content.size() - 1) break;
            if (pt.content[seg_idx].type == 1) 
                fixed_part += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2) 
                fixed_part += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 3) 
                fixed_part += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
        }
        if (pt.content.back().type == 1) a = &m.letters[m.FindLetter(pt.content.back())];
        else if (pt.content.back().type == 2) a = &m.digits[m.FindDigit(pt.content.back())];
        else if (pt.content.back().type == 3) a = &m.symbols[m.FindSymbol(pt.content.back())];
        N = pt.max_indices.back();
    }

    if (!a || N == 0) return;

    // 展平ordered_values
    vector<string>& values = a->ordered_values;
    vector<int> lengths(N);
    vector<int> offsets(N);
    size_t total_chars = 0;

    for (int i = 0; i < N; ++i) {
        lengths[i] = values[i].size();
        offsets[i] = total_chars;
        total_chars += lengths[i];
    }

    char* h_flatten = new char[total_chars];
    for (int i = 0; i < N; ++i) {
        memcpy(h_flatten + offsets[i], values[i].data(), lengths[i]);
    }

    // 设备内存分配
    char *d_flatten = nullptr, *d_output = nullptr;
    int *d_lengths = nullptr, *d_offsets = nullptr;
    char *d_fixed = nullptr;
    
    cudaMalloc(&d_flatten, total_chars);
    cudaMalloc(&d_lengths, N * sizeof(int));
    cudaMalloc(&d_offsets, N * sizeof(int));
    cudaMalloc(&d_output, N * MAX_PW_LEN);
    
    cudaMemcpy(d_flatten, h_flatten, total_chars, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, lengths.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // 配置核函数
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    if (pt.content.size() == 1) {
        generate_kernel_single<<<grid, block>>>(
            d_flatten, d_lengths, d_offsets, 
            d_output, MAX_PW_LEN, N
        );
    } else {
        cudaMalloc(&d_fixed, fixed_part.size() + 1);
        cudaMemcpy(d_fixed, fixed_part.c_str(), fixed_part.size() + 1, cudaMemcpyHostToDevice);
        
        generate_kernel_multi<<<grid, block>>>(
            d_fixed, fixed_part.size(),
            d_flatten, d_lengths, d_offsets,
            d_output, MAX_PW_LEN, N
        );
    }

    // 复制结果回主机
    char* h_output = new char[N * MAX_PW_LEN];
    cudaMemcpy(h_output, d_output, N * MAX_PW_LEN, cudaMemcpyDeviceToHost);

    // 添加生成的猜测
    for (int i = 0; i < N; ++i) {
        guesses.push_back(string(h_output + i * MAX_PW_LEN));
    }
    total_guesses += N;

    // 清理资源
    delete[] h_flatten;
    delete[] h_output;
    cudaFree(d_flatten);
    cudaFree(d_lengths);
    cudaFree(d_offsets);
    cudaFree(d_output);
    if (d_fixed) cudaFree(d_fixed);
}





//核函数实现
__global__ void generate_kernel_single(
    char* d_values, int* d_lengths, int* d_offsets, 
    char* d_output, int max_pw_len, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    char* output = d_output + idx * max_pw_len;
    int len = d_lengths[idx];
    char* value = d_values + d_offsets[idx];
    
    memcpy(output, value, len);
    output[len] = '\0';
}

__global__ void generate_kernel_multi(
    const char* d_guess, int guess_len,
    char* d_values, int* d_lengths, int* d_offsets,
    char* d_output, int max_pw_len, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    char* output = d_output + idx * max_pw_len;
    // 复制固定部分
    memcpy(output, d_guess, guess_len);
    // 添加动态部分
    int len = d_lengths[idx];
    char* value = d_values + d_offsets[idx];
    memcpy(output + guess_len, value, len);
    output[guess_len + len] = '\0';
}



