#pragma once
#include "simulator.hpp"

namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  
  // Pre-move all keys and values to HBM to avoid repeated moves in the loop
  for (size_t i = 0; i < keys.size(); ++i) {
    gpu_sim.MoveMatrixToGpuHbm(keys[i]);
    gpu_sim.MoveMatrixToGpuHbm(values[i]);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    size_t num_elements = i + 1;
    size_t d = keys[0]->GetColumnNum();

    // 1. Construct K_all [num_elements, d]
    Matrix* k_all = matrix_memory_allocator.Allocate("k_all");
    gpu_sim.Copy(keys[0], k_all, Position::kInGpuHbm);
    for (size_t j = 1; j < num_elements; ++j) {
      Matrix* temp_k = matrix_memory_allocator.Allocate("temp_k");
      gpu_sim.Concat(k_all, keys[j], temp_k, 0, Position::kInGpuHbm);
      gpu_sim.Copy(temp_k, k_all, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(temp_k);
    }

    // 2. Construct V_all [num_elements, d]
    Matrix* v_all = matrix_memory_allocator.Allocate("v_all");
    gpu_sim.Copy(values[0], v_all, Position::kInGpuHbm);
    for (size_t j = 1; j < num_elements; ++j) {
      Matrix* temp_v = matrix_memory_allocator.Allocate("temp_v");
      gpu_sim.Concat(v_all, values[j], temp_v, 0, Position::kInGpuHbm);
      gpu_sim.Copy(temp_v, v_all, Position::kInGpuHbm);
      gpu_sim.ReleaseMatrix(temp_v);
    }

    // 3. Compute S = Q * K_all^T
    // Q is [num_elements, d], K_all is [num_elements, d]
    Matrix* k_all_t = matrix_memory_allocator.Allocate("k_all_t");
    gpu_sim.Copy(k_all, k_all_t, Position::kInGpuHbm);
    gpu_sim.Transpose(k_all_t, Position::kInGpuHbm);

    Matrix* s = matrix_memory_allocator.Allocate("s");
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(k_all_t);
    gpu_sim.MatMul(current_query, k_all_t, s);

    // 4. Softmax(S) row-wise
    // S is [num_elements, num_elements]
    Matrix* softmax_s = matrix_memory_allocator.Allocate("softmax_s");
    
    for (size_t r = 0; r < num_elements; ++r) {
      Matrix* row = matrix_memory_allocator.Allocate("row");
      gpu_sim.GetRow(s, r, row, Position::kInSharedMemory);
      
      Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row");
      gpu_sim.MatExp(row, exp_row);
      
      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(exp_row, row_sum);
      
      Matrix* norm_row = matrix_memory_allocator.Allocate("norm_row");
      gpu_sim.MatDiv(exp_row, row_sum, norm_row);
      
      if (r == 0) {
        gpu_sim.Copy(norm_row, softmax_s, Position::kInSharedMemory);
        gpu_sim.Reshape(softmax_s, 1); 
      } else {
        Matrix* temp_softmax = matrix_memory_allocator.Allocate("temp_softmax");
        gpu_sim.Concat(softmax_s, norm_row, temp_softmax, 0, Position::kInSharedMemory);
        gpu_sim.Copy(temp_softmax, softmax_s, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(temp_softmax);
      }
      
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(norm_row);
    }

    // 5. Answer = Softmax(S) * V_all
    Matrix* answer = matrix_memory_allocator.Allocate("answer");
    gpu_sim.MoveMatrixToSharedMem(v_all);
    gpu_sim.MatMul(softmax_s, v_all, answer);
    
    // 6. Move Answer to HBM and commit
    gpu_sim.MoveMatrixToGpuHbm(answer);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);

    // Cleanup
    gpu_sim.ReleaseMatrix(k_all);
    gpu_sim.ReleaseMatrix(v_all);
    gpu_sim.ReleaseMatrix(k_all_t);
    gpu_sim.ReleaseMatrix(s);
    gpu_sim.ReleaseMatrix(softmax_s);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
