# \# Experiment 5: Neural Network Layers in CUDA

# 

# \## 1. Aim

# 

# To implement basic neural-network layers using CUDA and compare their execution performance on CPU and GPU.

# 

# \## 2. Hardware and Software Requirements

# 

# | Component | Specification |

# |---|---|

# | GPU | NVIDIA GeForce RTX 5060 Ti (16 GB) |

# | Operating System | Windows 11 |

# | CUDA Toolkit | 13.4.92 |

# | Compiler | NVIDIA NVCC |

# | IDE / Build Tools | Visual Studio 2026 Community |

# 

# \## 3. Introduction

# 

# CUDA (Compute Unified Device Architecture) is a parallel computing platform developed by NVIDIA. It allows programs to execute computations on an NVIDIA GPU.

# 

# In this experiment, CUDA is used to implement neural-network operations and compare their performance with CPU execution.

# 

# The experiment implements a two-layer neural-network forward pass.

# 

# \## 4. Operations Implemented

# 

# The following operations are implemented:

# 

# 1\. Matrix multiplication with bias

# 2\. ReLU activation function

# 3\. Sigmoid activation function

# 

# \### Forward-pass equations

# 

# ```text

# Z1 = X × W1 + b1

# A1 = ReLU(Z1)

# Z2 = A1 × W2 + b2

# Y  = Sigmoid(Z2)

# ```

# 

# \### Network dimensions

# 

# | Layer | Dimensions |

# |---|---|

# | Input | 256 × 128 |

# | Hidden Layer | 256 × 64 |

# | Output | 256 × 10 |

# 

# \*\*Note:\*\* This experiment performs forward propagation only. It does not perform network training or backpropagation.

# 

# \## 5. Project Structure

# 

# ```text

# Experiment5\_CUDA/

# │

# ├── neural\_network.cu

# ├── neural\_network.exe

# ├── README.md

# │

# ├── images/

# │   ├── cpu\_vs\_gpu\_end\_to\_end.png

# │   ├── gpu\_execution\_times.png

# │   └── speedup\_comparison.png

# │

# └── screenshots/

# &#x20;   ├── compilation.png

# &#x20;   ├── kernel\_tests.png

# &#x20;   └── final\_execution.png

# ```

# 

# \## 6. Compilation and Execution

# 

# Open Visual Studio 2026 x64 Developer Command Prompt.

# 

# Navigate to the experiment folder:

# 

# ```cmd

# cd /d "%USERPROFILE%\\Documents\\Experiment5\_CUDA"

# ```

# 

# Compile the CUDA program:

# 

# ```cmd

# nvcc -O2 neural\_network.cu -o neural\_network.exe

# ```

# 

# Execute the program:

# 

# ```cmd

# neural\_network.exe

# ```

# 

# \## 7. Correctness Verification

# 

# All three CUDA kernel tests passed.

# 

# | Test | Result |

# |---|---|

# | Matrix multiplication + bias | PASSED |

# | ReLU | PASSED |

# | Sigmoid | PASSED |

# | CPU and GPU output comparison | MATCH |

# | Maximum absolute output difference | 0.000000 |

# 

# The CPU and GPU outputs matched within the program's comparison tolerance.

# 

# \## 8. Performance Results

# 

# The experiment was executed three times.

# 

# \### Execution time comparison

# 

# | Run | CPU Compute (ms) | GPU Kernel-only (ms) | GPU End-to-end (ms) |

# |---|---:|---:|---:|

# | Run 1 | 0.962320 | 0.030702 | 0.134035 |

# | Run 2 | 0.943440 | 0.034179 | 0.126330 |

# | Run 3 | 1.023230 | 0.037722 | 0.132350 |

# | \*\*Average\*\* | \*\*0.976330\*\* | \*\*0.034201\*\* | \*\*0.130905\*\* |

# 

# \### Average speedup

# 

# | Measurement | Average Speedup |

# |---|---:|

# | GPU Kernel-only | \*\*28.69×\*\* |

# | GPU End-to-end | \*\*7.46×\*\* |

# 

# \## 9. Performance Graphs

# 

# \### 9.1 CPU vs GPU End-to-End Execution Time

# 

# This graph compares CPU compute time with GPU end-to-end execution time across three runs.

# 

# !\[CPU vs GPU End-to-End Execution Time](images/cpu\_vs\_gpu\_end\_to\_end.png)

# 

# \### 9.2 GPU Execution Time Across Three Runs

# 

# This graph compares GPU kernel-only execution time and GPU end-to-end execution time.

# 

# !\[GPU Execution Time](images/gpu\_execution\_times.png)

# 

# \### 9.3 CPU-to-GPU Speedup Comparison

# 

# This graph shows the kernel-only and end-to-end speedup obtained in each run.

# 

# !\[CPU-to-GPU Speedup](images/speedup\_comparison.png)

# 

# \## 10. Execution Screenshots

# 

# \### 10.1 Successful CUDA Compilation

# 

# The CUDA source file was compiled successfully using NVCC without compilation errors.

# 

# !\[Successful CUDA Compilation](screenshots/compilation.png)

# 

# \### 10.2 CUDA Kernel Testing

# 

# The individual CUDA kernel tests completed successfully.

# 

# !\[CUDA Kernel Tests](screenshots/kernel\_tests.png)

# 

# \### 10.3 Final Program Execution

# 

# The final output displays the GPU details, network dimensions, execution timings, speedups and CPU-GPU output comparison.

# 

# !\[Final Execution Output](screenshots/final\_execution.png)

# 

# \## 11. Observations

# 

# 1\. All three CUDA kernel tests passed successfully.

# 2\. The GPU execution time was lower than the CPU compute time for the tested workload.

# 3\. The average GPU kernel-only speedup was 28.69×.

# 4\. The average GPU end-to-end speedup was 7.46×.

# 5\. The maximum absolute output difference was 0.000000.

# 6\. CPU and GPU outputs matched within the specified tolerance.

# 

# \## 12. Conclusion

# 

# The neural-network layers were successfully implemented using CUDA on an NVIDIA GeForce RTX 5060 Ti GPU.

# 

# Matrix multiplication with bias, ReLU and Sigmoid operations were tested successfully. The CPU and GPU outputs matched with a maximum absolute difference of 0.000000.

# 

# Across three executions, the average GPU kernel-only speedup was 28.69× and the average GPU end-to-end speedup was 7.46× compared with CPU execution.

# 

# Therefore, the experiment demonstrates how CUDA can accelerate neural-network computations using GPU parallel processing.

# 

# \---

# \# Experiment 5: Neural Network Layers in CUDA

# 

# \*\*Author:\*\* Vandana A M  

# \*\*Branch:\*\* CSE (Artificial Intelligence)  

# \*\*Course:\*\* Parallel and GPU Computing (PGC)  

# \*\*Experiment:\*\* 5 – Neural Network Layers in CUDA

# 

# \*\*Technologies:\*\* CUDA, C++, NVIDIA GPU

