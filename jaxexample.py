# ----------------- Imports -----------------

"""
- jax.numpy (jnp): JAX’s optimised NumPy replacement, supporting GPU/TPU acceleration.
- grad: Automatic differentiation.
- jit: Just-in-time compilation for optimised function execution.
- vmap: Vectorisation for efficient parallel processing.
- pmap: Parallel mapping across multiple devices.
- lax: JAX's low-level API for operations that can be parallelised or require custom gradients.
- random: Pseudo-random number generation.
- atexit: Ensures the distributed system is shut down when the script exits.
"""
import os
import sys

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, pmap, lax, random as jrandom
import time
import logging
from contextlib import contextmanager

# Force JAX to prioritise GPU if available, fallback to CPU otherwise
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import atexit
atexit.register(lambda: jax.distributed.shutdown() if jax.process_count() > 1 else None)

# ----------------- JAX Distributed Setup -----------------

# def initialise_jax_distributed():
#     master_addr = os.environ.get("MASTER_ADDR", "localhost")
#     master_port = int(os.environ.get("MASTER_PORT", "12345"))
#     num_nodes = int(os.environ.get("SLURM_NNODES", "1"))
#     num_devices_per_node = jax.local_device_count()

#     # Calculate global rank based on SLURM environment variables (or fallback)
#     local_rank = int(os.environ.get("SLURM_PROCID", "0"))
#     node_rank = int(os.environ.get("SLURM_NODEID", "0"))
#     global_rank = node_rank * num_devices_per_node + local_rank

#     total_processes = num_nodes * num_devices_per_node
    
#     if total_processes > 1:
#         print("working")
#         jax.distributed.initialize(
#             coordinator_address=f"{master_addr}:{master_port}",
#             num_processes=total_processes,
#             process_id=global_rank,
#         )
#     print("done")
#     logging.info(f"Node {jax.process_index()} initialised with {jax.local_device_count()} devices.")

# if jax.device_count() > 1:
#     initialise_jax_distributed()




# ----------------- Setup -----------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Context Manager for Timed Blocks
@contextmanager
def timed_block(name: str):
    start = time.time()
    yield
    duration = time.time() - start
    logging.info(f"{name} took {duration:.6f} seconds")

# Test Results Tracking
TEST_RESULTS = []

def track_result(test_name: str, passed: bool, error_message: str = None):
    result = {
        "name": test_name,
        "passed": passed,
        "error": error_message,
    }
    TEST_RESULTS.append(result)

# ----------------- Device Info -----------------

def check_devices() -> list:
    logging.info(f"\nDevice Info:")
    devices = jax.devices()
    # logging.info(f"Total Devices: {len(devices)}")
    logging.info(f"Total Devices: {len(devices)} across {jax.process_count()} nodes")
    device_types = {device.device_kind.lower() for device in devices}
    logging.info(f"Device Types: {device_types}")
    
    if "gpu" in device_types:
        logging.info(f"GPU is available! Number of devices: {len(devices)}")
    elif "tpu" in device_types:
        logging.info(f"TPU is available! Number of devices: {len(devices)}")
    else:
        logging.info("No GPU or TPU found. Running on CPU.")

    return devices

# ----------------- Basic Operations -----------------

def test_basic_operations() -> None:
    try:
        logging.info("Testing Basic Operations")
        a = jnp.arange(1, 9)
        b = jnp.arange(9, 17)
        logging.info(f"Sum: {jnp.add(a, b)}")
        logging.info(f"Elementwise Multiply: {jnp.multiply(a, b)}")
        logging.info(f"Dot Product: {jnp.dot(a, b)}")
        track_result("Basic Operations", True)
    except Exception as e:
        track_result("Basic Operations", False, str(e))
        logging.error(f"Basic Operations Error: {e}")

# ----------------- JIT Compilation -----------------

def test_jit_compilation() -> None:
    try:
        logging.info("Testing JIT Compilation")
        @jit
        def square(x: jnp.ndarray) -> jnp.ndarray:
            return x ** 2
        with timed_block("JIT Function Execution"):
            square(jnp.arange(1000000)).block_until_ready()
        track_result("JIT Compilation", True)
    except Exception as e:
        track_result("JIT Compilation", False, str(e))

# ----------------- Automatic Differentiation -----------------

def test_autodiff() -> None:
    try:
        logging.info("Testing Automatic Differentiation")
        f = lambda x: jnp.sin(x) ** 2 + jnp.cos(x) ** 2
        df = grad(f)
        logging.info(f"f(0.5) = {f(0.5)}, f'(0.5) = {df(0.5)}")
        track_result("Automatic Differentiation", True)
    except Exception as e:
        track_result("Automatic Differentiation", False, str(e))
        logging.error(f"Automatic Differentiation Error: {e}")

# ----------------- Vectorisation -----------------

def test_vectorisation() -> None:
    try:
        logging.info("Testing Vectorisation")
        square_plus_one = vmap(lambda x: x ** 2 + 1)
        vec_input = jnp.arange(1, 9)
        output = square_plus_one(vec_input)
        assert output.shape == vec_input.shape, "Shape mismatch!"
        logging.info(f"Vectorised Output: {output}")
        track_result("Vectorisation", True)
    except Exception as e:
        track_result("Vectorisation", False, str(e))
        logging.error(f"Vectorisation Error: {e}")

    
# ----------------- Array Manipulation -----------------

def test_array_manipulation() -> None:
    try:
        logging.info("Testing Array Manipulation")
        arr = jnp.arange(16).reshape(4, 4)
        logging.info(f"Original Array:\n{arr}")
        logging.info(f"Sliced Array (1:3, 1:3):\n{arr[1:3, 1:3]}")
        logging.info(f"Transposed Array:\n{arr.T}")
        track_result("Array Manipulation", True)
    except Exception as e:
        track_result("Array Manipulation", False, str(e))
        logging.error(f"Array Manipulation Error: {e}")

# ----------------- Linear Algebra -----------------

def test_linear_algebra() -> None:
    logging.info("Testing Linear Algebra")
    matrix = jnp.array([[1, 2], [3, 4]])
    try:
        logging.info(f"Matrix Inverse:\n{jnp.linalg.inv(matrix)}")
    except jnp.linalg.LinAlgError as e:
        logging.error(f"Matrix inversion error: {e}")
    logging.info(f"Matrix Determinant: {jnp.linalg.det(matrix)}")

# ----------------- Random Number Generation -----------------

def test_random_number_generation() -> None:
    try:
        logging.info("Testing Random Number Generation")
        key = jrandom.PRNGKey(42)
        rand_array = jrandom.normal(key, (3, 3))
        logging.info(f"Random Array:\n{rand_array}")
        track_result("Random Number Generation", True)
    except Exception as e:
        track_result("Random Number Generation", False, str(e))
        logging.error(f"Random Number Generation Error: {e}")

# ----------------- Parallel Computation -----------------

def test_parallel_computation(devices: list) -> None:
    """
    Test parallel computation using pmap or vmap based on the number of devices.
    """
    try:
        logging.info("Testing Parallel Computation")
        num_devices = len(devices)
        xs = jnp.arange(num_devices * 2).reshape((num_devices, 2))
        if num_devices > 1:
            parallel_add = pmap(lambda x: x + 1.0)
            logging.info(f"pmap result: {parallel_add(xs)}")
        else:
            vectorised_add = vmap(lambda x: x + 1.0)
            logging.info(f"vmap result (fallback): {vectorised_add(xs)}")
        track_result("Parallel Computation", True)
    except Exception as e:
        track_result("Parallel Computation", False, str(e))
        logging.error(f"Parallel Computation Error: {e}")

# ----------------- Distributed Computing -----------------

def test_distributed_computing(devices: list) -> None:
    try:
        logging.info("Testing Distributed Computing")
        num_devices = len(devices)
        xs = jnp.arange(num_devices * 2, dtype=jnp.float32).reshape((num_devices, 2))
        if num_devices > 1:
            def distributed_mean(x):
                return lax.pmean(x, axis_name="i")
            # Ensure axis_name is globally consistent across nodes
            result = pmap(distributed_mean, axis_name="i")(xs)
            logging.info(f"Distributed Mean: {result}")
        else:
            logging.info(f"Local Mean (single device): {jnp.mean(xs, axis=0)}")
        track_result("Distributed Computing", True)
    except Exception as e:
        track_result("Distributed Computing", False, str(e))

# ----------------- Summary of Results -----------------

def print_test_summary():
    logging.info("\n=== Test Summary ===")
    for result in TEST_RESULTS:
        status = "PASSED" if result["passed"] else "FAILED"
        error_info = f" - {result['error']}" if result['error'] else ""
        logging.info(f"{result['name']}: {status}{error_info}")

# ----------------- Main Test Runner -----------------

def run_all_tests() -> None:
    devices = check_devices()
    test_basic_operations()
    test_jit_compilation()
    test_autodiff()
    test_vectorisation()
    test_array_manipulation()
    test_linear_algebra()
    test_random_number_generation()
    test_parallel_computation(devices)
    test_distributed_computing(devices)
    print_test_summary()
    logging.info("All tests completed.")

if __name__ == "__main__":
    run_all_tests()