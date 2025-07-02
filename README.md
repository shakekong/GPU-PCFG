# GPU-PCFG

#guessing_base.cu为基础要求实现

#guessing_serial.cpp为原串行代码

#guessing_best.cu为进阶要求3和1结合考虑的代码，同时也是表现最好的版本，通过调整batch_size，可以观察cracked值的变化

#guessing.cu为单独的进阶要求3的代码

#correctness_guess和PCFG文件中，如果需要运行代码，请将这两个文件中并行部分取消注释，并注释掉相应的串行部分

#并行部分：const int BATCH_SIZE = 4; // 可调
          q.PopNextBatch(BATCH_SIZE);
          class PriorityQueue的私有成员变量以及函数 BatchGenerate 和 PopNextBatch


#串行部分：q.PopNext(); //串行

