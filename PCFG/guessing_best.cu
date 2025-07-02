#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
using namespace std;


// 添加阈值常量
#define MIN_GPU_SIZE 100000 // 小于此数量的PT用CPU处理
#define MAX_PW_LEN 128


__global__ void generate_kernel_batch(
    char* d_fixed, int* d_fixed_len, int fixed_stride,
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

__global__ void generate_kernel_batch(
    char* d_fixed, int* d_fixed_len, int fixed_stride,
    char* d_values, int* d_lengths, int* d_offsets,
    char* d_output, int max_pw_len, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    char* output = d_output + idx * max_pw_len;
    char* fixed = d_fixed + idx * fixed_stride;
    int fixed_len = d_fixed_len[idx];
    for (int i = 0; i < fixed_len; ++i) output[i] = fixed[i];

    int len = d_lengths[idx];
    char* value = d_values + d_offsets[idx];
    for (int i = 0; i < len; ++i) output[fixed_len + i] = value[i];
    output[fixed_len + len] = '\0';
}

void PriorityQueue::BatchGenerate(const vector<PT>& pts) {
    // 1. 统计所有PT的任务总数，并分流
    struct TaskInfo {
        int pt_idx;
        int N;
        string fixed_part;
        segment* last_seg;
        bool is_single;
    };
    vector<TaskInfo> cpu_tasks, gpu_tasks;
    vector<int> Ns;
    vector<string> fixed_parts;
    vector<segment*> last_segments;
    int max_fixed_len = 0;

    for (size_t i = 0; i < pts.size(); ++i) {
        const auto& pt = pts[i];
        bool single = (pt.content.size() == 1);
        segment* a = nullptr;
        int N = 0;
        string fixed_part;
        if (single) {
            if (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
            else if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
            else if (pt.content[0].type == 3) a = &m.symbols[m.FindSymbol(pt.content[0])];
            N = pt.max_indices[0];
            fixed_part = "";
        } else {
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
        if ((int)fixed_part.size() > max_fixed_len) max_fixed_len = fixed_part.size();

        TaskInfo info = {int(i), N, fixed_part, a, single};
        if (N < MIN_GPU_SIZE)
            cpu_tasks.push_back(info);
        else
            gpu_tasks.push_back(info);

        Ns.push_back(N);
        fixed_parts.push_back(fixed_part);
        last_segments.push_back(a);
    }

    // 2. 先处理CPU任务
    for (const auto& task : cpu_tasks) {
        if (!task.last_seg || task.N == 0) continue;
        if (task.is_single) {
            for (int i = 0; i < task.N; ++i)
                guesses.push_back(task.last_seg->ordered_values[i]);
        } else {
            for (int i = 0; i < task.N; ++i)
                guesses.push_back(task.fixed_part + task.last_seg->ordered_values[i]);
        }
        total_guesses += task.N;
    }

    // 3. 再处理GPU任务（合批送入）
    int total_gpu_tasks = 0;
    for (const auto& task : gpu_tasks) total_gpu_tasks += task.N;
    if (total_gpu_tasks == 0) return;

    // 合批准备数据
    vector<string> all_fixed(total_gpu_tasks);
    vector<int> all_fixed_len(total_gpu_tasks);
    vector<string> all_values(total_gpu_tasks);
    vector<int> all_lengths(total_gpu_tasks);
    vector<int> all_offsets(total_gpu_tasks);
    int offset = 0, task_idx = 0;
    size_t total_chars = 0;
    for (const auto& task : gpu_tasks) {
        for (int j = 0; j < task.N; ++j) {
            all_fixed[task_idx] = task.fixed_part;
            all_fixed_len[task_idx] = task.fixed_part.size();
            all_values[task_idx] = task.last_seg->ordered_values[j];
            all_lengths[task_idx] = task.last_seg->ordered_values[j].size();
            all_offsets[task_idx] = total_chars;
            total_chars += all_lengths[task_idx];
            task_idx++;
        }
    }

    // flatten values
    char* h_flatten = new char[total_chars];
    size_t curr = 0;
    for (int i = 0; i < total_gpu_tasks; ++i) {
        memcpy(h_flatten + curr, all_values[i].data(), all_lengths[i]);
        curr += all_lengths[i];
    }

    // flatten fixed
    char* h_fixed = new char[total_gpu_tasks * max_fixed_len];
    memset(h_fixed, 0, total_gpu_tasks * max_fixed_len);
    for (int i = 0; i < total_gpu_tasks; ++i) {
        if (all_fixed[i].size() > 0)
            memcpy(h_fixed + i * max_fixed_len, all_fixed[i].data(), all_fixed[i].size());
    }

    // 分配显存
    char *d_flatten = nullptr, *d_output = nullptr, *d_fixed = nullptr;
    int *d_lengths = nullptr, *d_offsets = nullptr, *d_fixed_len = nullptr;
    cudaMalloc(&d_flatten, total_chars);
    cudaMalloc(&d_output, total_gpu_tasks * MAX_PW_LEN);
    cudaMalloc(&d_lengths, total_gpu_tasks * sizeof(int));
    cudaMalloc(&d_offsets, total_gpu_tasks * sizeof(int));
    cudaMalloc(&d_fixed, total_gpu_tasks * max_fixed_len);
    cudaMalloc(&d_fixed_len, total_gpu_tasks * sizeof(int));

    // 拷贝数据到设备
    cudaMemcpy(d_flatten, h_flatten, total_chars, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, all_lengths.data(), total_gpu_tasks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, all_offsets.data(), total_gpu_tasks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fixed, h_fixed, total_gpu_tasks * max_fixed_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fixed_len, all_fixed_len.data(), total_gpu_tasks * sizeof(int), cudaMemcpyHostToDevice);

    // kernel launch
    dim3 block(256);
    dim3 grid((total_gpu_tasks + block.x - 1) / block.x);
    generate_kernel_batch<<<grid, block>>>(
        d_fixed, d_fixed_len, max_fixed_len,
        d_flatten, d_lengths, d_offsets,
        d_output, MAX_PW_LEN, total_gpu_tasks
    );
    cudaDeviceSynchronize();

    // 获取结果
    char* h_output = new char[total_gpu_tasks * MAX_PW_LEN];
    cudaMemcpy(h_output, d_output, total_gpu_tasks * MAX_PW_LEN, cudaMemcpyDeviceToHost);

    // 添加结果
    for (int i = 0; i < total_gpu_tasks; ++i) {
        guesses.push_back(string(h_output + i * MAX_PW_LEN));
    }
    total_guesses += total_gpu_tasks;

    // 清理
    delete[] h_flatten;
    delete[] h_fixed;
    delete[] h_output;
    cudaFree(d_flatten);
    cudaFree(d_output);
    cudaFree(d_lengths);
    cudaFree(d_offsets);
    cudaFree(d_fixed);
    cudaFree(d_fixed_len);
}

void PriorityQueue::PopNextBatch(int batch_size) {
    int actual_batch = min(batch_size, (int)priority.size());
    vector<PT> batch_pts(priority.begin(), priority.begin() + actual_batch);

    // 合批生成
    BatchGenerate(batch_pts);

    // 统一收集所有新 PT
    vector<PT> all_new_pts;
    for (auto& pt : batch_pts) {
        vector<PT> new_pts = pt.NewPTs();
        all_new_pts.insert(all_new_pts.end(), new_pts.begin(), new_pts.end());
    }

    // 批量插入新 PT（可优化为排序后合并）
    for (PT& pt : all_new_pts) {
        CalProb(pt);
        auto iter = priority.begin();
        while (iter != priority.end() && pt.prob < iter->prob) ++iter;
        priority.insert(iter, pt);
    }

    // 删除已处理的 PT
    priority.erase(priority.begin(), priority.begin() + actual_batch);
}
