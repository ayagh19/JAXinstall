# **JAX Install for Intel GPUs**  

This repository contains convenient bash scripts to install JAX on Intel GPUs with a pip installation, using the Intel® Extension for OpenXLA plug-in. 

## **Installation** 
To install JAX with Intel GPU support, simply run:  
```bash
./install.sh  
./setup.sh  
```

## **Verification** 
To verify that JAX is correctly installed and the Intel GPU is detected, run:
```bash
python -c "import jax; print(jax.devices())"
```
It should give somthing like
```bash
[sycl(id=0), sycl(id=1)]
```

## **JAX Features Test Script** 
It also contains a sample script `jaxexample.py` to verify that some key JAX features are functioning correctly:
1. **Device Info Check** - Confirming GPU is being used.  
2. **Basic JAX Operations** - Element-wise operations, random number generation.  
3. **JIT Compilation** - Speedup using `jax.jit`.  
4. **Automatic Differentiation** - Using `jax.grad` and `jax.vjp`.  
5. **Parallel Computation** - Using `jax.pmap`.  
6. **Vectorisation** - apply functions over entire arrays without for-loops using `jax.vmap`.
7. **Array Manipulation** - Advanced indexing, slicing, and reshaping.  
8. **Linear Algebra** - Matrix operations.  
9. **Random Number Generation** - Using the new `jax.random` PRNG.  
10. **Distributed Computing** - Basic test for `jax.lax.pmean` and `pmap` over multiple devices.  