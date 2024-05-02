# binaryfields
simulating max average load of linear functions


the average max load of all hash functions approaches O(log n /log log n) for large n by a straightforward application of chernoff. 

For the case of linear functions over finite fields, the same proof fails bc e.g. f(a), f(b) fixes f(a+b) 

i leave a correct proof to men with glasses, let's just simulate it.

consider concretely functions from F_2^{2n} to F_2^n

There are 2^{2 * n * n} such functions. And 2^{2n} vectors to try as inputs. this is quickly intractable for n > 4 or 5. So we will pick some quantity of random vectors over F_2^{2n} and multiply them through and count the most common one.

Let's say m matrices each tested with v random vectors. To do this on the GPU, we should initialize M matrices and m * v random vectors in device memory.

First we divide initialize these random vectors in parallel, storing the result in global memory. With matrices, we can likely store them in smem for n < 100 to provide some speedup. Each block gets one matrix, which then executes the matrix vector multiplication in parallel. 


