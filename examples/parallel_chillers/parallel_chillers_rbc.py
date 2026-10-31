"""Sequential rule-based controller, Eqs. (5)–(6).

J. Boldocky, C. Faulkner, E. Michael, M. Gulan, A. Tuor, and J. Drgona,
Data Center Chiller Plant Optimization via Mixed-Integer Nonlinear
Differentiable Predictive Control, 2026. https://arxiv.org/html/2603.28323v1
Implements the paper's rule-based policy in NumPy. Actuator conventions were
checked against pnnl/MI-DPC revision 31ce8e6dd68c5d27bb4ac202d44fe82c3c7fbb54.
Model implementation and data: https://github.com/pnnl/MI-DPC
"""
import numpy as np


class RuleBasedController:
    def __init__(self, model):
        self.model = model
        self.active = 1

    def __call__(self, state):
        c = self.model.config
        order = list(range(c.M))
        if c.permanent_on_unit is not None:
            order.remove(c.permanent_on_unit)
            order.insert(0, c.permanent_on_unit)
        switches = np.zeros(c.M)
        switches[order[:self.active]] = 1
        continuous = np.r_[10.0*switches, np.full(c.M, 9.0)]
        cooling = self.model.evaluate(state, continuous, switches)[0]
        plr = sum(cooling)/(c.capacity*self.active)
        if plr > 0.6:
            self.active += 1
        if plr < 0.15:
            self.active -= 1
        self.active = min(c.M, max(1, self.active))
        return continuous, switches
