Debugging in Wave
=================

To use printf-style debugging in wave, you can use the ``debug_log`` wave op.

The ``debug_log`` op's first argument is a register value that you want to see.  Next it has an optional ``label`` argument, a string.

Suppose we have this basic GEMM kernel:

.. code-block:: python

    @tkw.wave(constraints)
    def gemm(
        a: Memory[M, K, ADDRESS_SPACE, f16],
        b: Memory[N, K, ADDRESS_SPACE, f16],
        c: Memory[M, N, GLOBAL_ADDRESS_SPACE, f32],
    ):
        c_reg = Register[M, N, f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)

            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # Store the final result to C
        tkw.write(repeat, c)

We can sprinkle in some debug_logs to get a picture of what is going on inside.

.. code-block:: python

    a_reg = tkw.read(a)
    tkw.debug_log(a_reg, label="a_reg", printer=print)

Note that the printer function (which defaults to `print`) will receive 2 arguments: the label, and the tensor data.
This will print out a global view of the contents of ``a_reg``, where “global view” means that it will print an entire MxK matrix based on the ``a_reg`` value from each wave.
The ``debug_log`` op works by writing to global memory, where the global memory is automatically added to the kernel signature and launch arguments.
Since ``a_reg`` is just read from the ``a`` memory, with an identity mapping, the result of the debug_log is equal to the ``a`` matrix.

Let's print something more interesting, let's print the ``acc`` register:

.. code-block:: python

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)

            acc = tkw.mma(a_reg, b_reg, acc)
            tkw.debug_log(acc, label="acc", printer=print)
            return acc

Because the debug log represents a write to global memory, each iteration of the loop will overwrite the previous value.
In the end the ``acc`` log will have the same value as the final output ``c``.
To see intermediate values, we can use the ``extra_iteration_dimensions`` argument.

.. code-block:: python

    tkw.debug_log(
        acc,
        label="acc",
        extra_iteration_dimensions=[(tkl.sym.iter, K, 4)],
        printer=print
        )

This new ``acc`` log will be a ``4xMxN`` tensor, where the outer size-4 dimension is named ``iter``.
Each iteration along the ``K`` dimension by the ``iterate`` form will write to the next slot in that outer dimension, with a max of 4.
If you give a number of iterations that is too small, later iterations will overwrite the final slot.

Printing each tensor on the terminal may not be very helpful.
The view of the tensor is probably truncated, and you may want to compare multiple tensors to understand what is happening.
Instead of using the ``printer`` argument, you can use a ``handler`` argument that will receive a dictionary with all of the debug log tensors, along with some metadata about them.
In particular, you can use this with the ``html_viewer`` function to generate a web page with an interactive view of the debug log tensors.

.. code-block:: python

    from wave_lang.debugging.html_viewer import html_viewer
    tkw.debug_log(a_reg, handler=html_viewer)
    tkw.debug_log(acc, label="acc", extra_iteration_dimensions=[(tkl.sym.iter, K, 4)])

Due to the use of ``handler=html_viewer``, a ``debug-view.html`` file will be generated in the current directory of the process that calls the gemm kernel.
Open this file in a web browser, and you can see (a view of) all tensors (in this case, for ``a_reg`` and for ``acc``).
Each tensor is displayed with at most 2 dimensions at once, but you can interactively change which dimensions are viewed, and sweep over different fixed values for other dimensions.
The view for each tensor in the html view has an editable text box listing the dimensions, eg. ``iter=0, M, N`` for the ``acc`` log above.
Sweep over iterations by changing the number after ``iter=``, or fix the ``M`` to row 5 by setting ``M=5``, or transpose the view by reordering the dimensions.
For example, writing ``iter, M=3, N``, you will see a matrix where each row represents row ``M=3`` of the ``MxN`` ``acc`` matrix, except that row 0 is for iteration 0, row 1 is for iteration 1, etc.
Or ``M, iter, N=0`` will show each column as column ``N=0``, sweeping left-to-right over the iterations.
