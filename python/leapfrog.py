import numpy as np
from tqdm import tqdm


def _one_step(dq_h, dp_h, q, p, step):
    curr_p = p
    curr_q = q
    curr_p = curr_p - step / 2 * dq_h(curr_q, curr_p)
    curr_q = curr_q + step * dp_h(curr_q, curr_p)
    curr_p = curr_p - step / 2 * dq_h(curr_q, curr_p)
    return curr_q, curr_p


def explicit_leapfrog(dq_h, dp_h, q0, p0, step, t_stop=1):
    """
    Simulates the given Hamiltonian system with the leapfrog method as described in Equation (3.35) in
    [Owhadi2020]. Returns the flows at time t_stop
    :param t_stop: stop time
    :param p0: initial momentum
    :param q0: initial position
    :param step: size of step
    :param dq_h: partial derivative of Hamiltonian w.r.t q
    :param dp_h: partial derivative of Hamiltonian w.r.t p
    :return: flows q and p at time t_stop
    """
    num_steps = int(1 / step * t_stop) + 1
    curr_q = q0
    curr_p = p0
    for _ in tqdm(range(1, num_steps), disable=True):
        curr_q, curr_p = _one_step(dq_h, dp_h, curr_q, curr_p, step)
    return curr_q, curr_p


def simulate_flow(dq_h, dp_h, q0, p0, step, t_stop=1):
    """
    Simulates the given Hamiltonian system with the leapfrog method as described in Equation (3.35) in
    [Owhadi2020]. Returns the flows over time.
    :param t_stop: stop time
    :param p0: initial momentum
    :param q0: initial position
    :param step: size of step
    :param dq_h: partial derivative of Hamiltonian w.r.t q
    :param dp_h: partial derivative of Hamiltonian w.r.t p
    :return: flows q and p
    """
    num_steps = int(1 / step * t_stop) + 1

    q = np.zeros((num_steps, *q0.shape))
    p = np.zeros((num_steps, *p0.shape))
    q[0] = q0.numpy()
    p[0] = p0.numpy()

    curr_q = q0
    curr_p = p0
    for i in tqdm(range(1, num_steps)):
        curr_q, curr_p = _one_step(dq_h, dp_h, curr_q, curr_p, step)
        p[i] = curr_p.numpy()
        q[i] = curr_q.numpy()
    return q, p
