# Incremental Support Vector Machine on FPGA

This system trains a standard ε-insensitive Support Vector Machine for Regression (ε-SVR) incrementally on FPGA.

The system supports both incremental training (when a new data sample joins the training dataset) as well as decremental training (when an existing data sample is removed from the training dataset). The datapath is parallelisd to improve performance. The parallelism factor is adjustable to make the design scalable and to make full use of the available FPGA resources.

We have a paper describing the details about this project:
 * Shengjia Shao, Oskar Mencer and Wayne Luk, "Dataflow Design for Optimal Incremental SVM Training," in The 2016 International Conference on Field-Programmable Technology (FPT '16). (to appear)
 
### System Requirements

1. CentOS 5/6 (x86_64) or Red Hat Enterprise Linux 5/6 (x86_64).
2. GNU Make 3.80+ & GCC 3.4+ & Intel Compiler 12.1+ (optional).
3. Altera Quartus 13.0.
4. Maxeler MaxCompiler version 2014.2+.
5. Maxeler OS version 2014.2+.
6. The Maxeler platform with Maia dataflow engine (Altera Stratix V GS 5SGSD8 FPGA)


### Example Applications

Stock Mid-Price prediction (`build/data9970.txt`) using ε-SVR with 16 features.

### How to Use

1. Customising the SVM:
 - `src/Def.maxj`
 
    Here we set the parallelism, window size, number representation, etc. These settings will have significant influence on the hardware resource usage (and whether the design can fit into the FPGA). More details to be added.
    
 - `src/SVMKernel.maxj`
 
    Set the RBFKernel() function to the kernel to use. Currently the Gaussian RBF kernel is used.

2. Customising the Makefiles:

 - `build/Makefile.common`
 
   In '---- Compiler settings ----' section, we set the C compilers to use via the $(CXX) and $(CC) variables. Set them to icc for Intel C Compiler or gcc for GCC C Compiler based on your developing environment.

 - `build/Makefile.Maia.hardware`
 
   Here we set the configuration of the Maxeler platform. If you use the infiniband based MPC-X system, set the MPCX option to 'true', and specify the SLIC_CONF variable according to your system configuration. If you are using PCI-E based system, set the MPCX option to 'false' and comment out the SLIC_CONF line.
   
3. Setting the number of threads to use when building hardware:

  - `src/SVMManager.maxj`
  
    Use Config.setMPPRParallelism() to set the number of threads to use when building hardware. Each thread will need ~20GB memory. Therefore the number of threads to use depends on the amount of RAM available in the computer. For example, if you want to use 4 threads, set Config.setMPPRParallelism(4).

4. (Optional) set the cost table search range when building hardware:

  - `src/SVMManager.maxj`
  
     Set the argument in Config.setMPPRCostTableSearchRange(,) to the start number and end number of cost table to try.

3. Build the System

 * Simulation:
 	* `make runsim`
 * Build hardware:
 	* `make build`
 * Run hardware:
 	* `make run` 

