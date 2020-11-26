import numpy as np
from tqdm import tqdm


def explicit_leapfrog(dq_h, dp_h, q0, p0, step, t_stop=1):
    """
    Simulates the given Hamiltonian system with the leapfrog method as described in Equation (3.35) in
    [Owhadi2020].
    :param t_stop: stop time
    :param p0: initial momentum
    :param q0: initial position
    :param step: size of step
    :param dq_h: partial derivative of Hamiltonian w.r.t q
    :param dp_h: partial derivative of Hamiltonian w.r.t p
    :return: flows q and p
    """
    q0 = np.asarray(q0)
    p0 = np.asarray(p0)
    num_steps = int(1 / step) * t_stop
    q = np.zeros((num_steps, *q0.shape))
    p = np.zeros((num_steps, *p0.shape))
    q[0] = q0
    p[0] = p0
    for i in range(1, num_steps):
        curr_q = q[i - 1]
        curr_p = p[i - 1]
        curr_p = curr_p - step / 2 * dq_h(curr_q, curr_p)
        curr_q = curr_q + step * dp_h(curr_q, curr_p)
        curr_p = curr_p - step / 2 * dq_h(curr_q, curr_p)
        p[i] = curr_p
        q[i] = curr_q
    return q, p
