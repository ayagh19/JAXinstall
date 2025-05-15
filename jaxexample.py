import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, grad, jit, vmap, pmap, lax
import numpy as np
import time

# to do 
# set up jax to use devices on more than one node


# ----------------- DEVICE INFO CHECK -----------------

def check_devices():
    print("\nDevice Info:")
    devices = jax.devices()
    device_types = {device.device_kind.lower() for device in devices}
    print(f"Total Devices: {len(devices)}")
    print(f"Device Types: {device_types}")
    
    if "gpu" in device_types:
        print(f"GPU is available! Number of devices: {len(devices)}")
    elif "tpu" in device_types:
        print(f"TPU is available! Number of devices: {len(devices)}")
    else:
        print("No GPU or TPU found. Running on CPU.")
    
    return devices

# ----------------- BASIC OPERATIONS -----------------

def basic_operations():
    print("\nBasic JAX Operations:")
    a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    b = jnp.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    print("Sum:", jnp.add(a, b))
    print("Elementwise Multiply:", jnp.multiply(a, b))
    print("Dot Product:", jnp.dot(a, b))

# ----------------- JIT COMPILATION -----------------

@jit
def timeit(x):
    return x ** 2

def jit_test():
    print("\nJIT Compilation Test:")
    x = jnp.arange(1000000)
    start = time.time()
    timeit(x).block_until_ready()
    print(f"JIT function execution time: {time.time() - start:.6f} seconds")

# ----------------- AUTOMATIC DIFFERENTIATION -----------------

def autodiff_test():
    print("\nAutomatic Differentiation:")
    f = lambda x: jnp.sin(x) ** 2 + jnp.cos(x) ** 2
    df = grad(f)
    print(f"f(0.5) = {f(0.5)}, f'(0.5) = {df(0.5)}")
    
    v, vjp_fn = jax.vjp(f, 0.5)
    print(f"VJP at 0.5: {v}, VJP Function: {vjp_fn(1.0)}")

# ----------------- VECTORISATION -----------------

def vectorization_test():
    print("\nVectorisation with vmap:")
    def square_plus_one(x):
        return x ** 2 + 1
    
    batched_square = vmap(square_plus_one)
    vec_input = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    print("Vectorised Output:", batched_square(vec_input))


# ----------------- ARRAY MANIPULATION -----------------

def array_manipulation_test():
    print("\nAdvanced Array Manipulations:")
    arr = jnp.arange(16).reshape(4, 4)
    print("Sliced Array:", arr[1:3, 1:3]) 
    print("Transposed Array:", arr.T)

# ----------------- LINEAR ALGEBRA -----------------

def linear_algebra_test():
    print("\nLinear Algebra:")
    matrix = jnp.array([[1, 2], [3, 4]])
    print("Matrix Inverse:", jnp.linalg.inv(matrix))
    print("Matrix Determinant:", jnp.linalg.det(matrix))

# ----------------- RANDOM NUMBER GENERATION -----------------

def random_number_test():
    print("\nRandom Number Generation:")
    key = jrandom.PRNGKey(42)
    rand_array = jrandom.normal(key, (3, 3))
    print("Random Array:", rand_array)

# ----------------- PARALLEL COMPUTATION -----------------

def parallel_computation_test():
    print("\nParallel Computation Test:")
    num_devices = len(jax.devices())
    xs = jnp.arange(num_devices * 2).reshape((num_devices, 2))
    
    # Use pmap if more than one device is available, otherwise use vmap as a single-device fallback
    if num_devices > 1:
        parallel_add = pmap(lambda x: x + 1.0)
        print("pmap result:", parallel_add(xs))
    else:
        print("Only one device found, using vmap instead.")
        vectorised_add = vmap(lambda x: x + 1.0)
        print("vmap result:", vectorised_add(xs))

# ----------------- DISTRIBUTED COMPUTING -----------------

def distributed_computing_test():
    print("\nDistributed Computing Test:")
    num_devices = len(jax.devices())
    xs = jnp.arange(num_devices * 2).reshape((num_devices, 2))
    if num_devices > 1:
        def distributed_mean(x):
            return lax.pmean(x, axis_name="i")
        print("Distributed Mean:", pmap(distributed_mean, axis_name="i")(xs))
    else:
        print("Only one device found, skipping pmap. Running local mean instead.")
        print("Local Mean:", jnp.mean(xs, axis=0))

# ----------------- MAIN FUNCTION -----------------

def main():
    print("\n=== JAX Test Script ===\n")
    check_devices()
    basic_operations()
    jit_test()
    autodiff_test()
    vectorization_test()
    array_manipulation_test()
    linear_algebra_test()
    random_number_test()

    parallel_computation_test()
    distributed_computing_test()


if __name__ == "__main__":
    main()