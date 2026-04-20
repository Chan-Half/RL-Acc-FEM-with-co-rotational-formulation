This package requires gcc and g++ version 8 or lower (requirements for PyCUDA).

See the instructions in the .txt file for operation details.

The current version uses the co-rotational coordinate method . The problem being addressed is spurious high-frequency vibration.

September 6, 2025: The spurious high-frequency components have been solved using the HHT method. Now, we are attempting to use displacement as the unknown variable instead of acceleration.

We are also exploring the effects of different solution methods, such as quasi-static methods. (Currently, we have found that osqp and cvxopt produce different results for the same problem with and without a solution.)

September 17, 2025: Using displacement as the unknown variable instead of acceleration has been completed. We are exploring the effects of different solution methods, such as quasi-static methods, and methods to prevent energy growth.
