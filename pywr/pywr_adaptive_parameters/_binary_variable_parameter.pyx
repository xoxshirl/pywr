from pywr.parameters._parameters cimport Parameter, IndexParameter
from pywr.parameters._parameters import Parameter, IndexParameter, load_parameter
from pywr._core cimport Timestep
cimport numpy as np
import numpy as np


cdef class BinaryVariableParameter(IndexParameter):
    """ Parameter to be used as a binary switch between two other parameters.
    This parameter is intended to be used as a variable (i.e. is_variable=True). It will
    switch between the `enabled_parameter` and `disabled_parameter` when the state is
    true or false respectively.
    Parameters
    ----------
    model : `pywr.Model`
    enabled_parameter : `Parameter`
        The parameter to use when the value is true (1).
    disabled_parameter : `Parameter`
        The parameter to use when the value is false (0).
    value : int (0 or 1)
        The initial state of the parameter.

    """
    def __init__(self, model, enabled_parameter, disabled_parameter, value=0, **kwargs):
        super(BinaryVariableParameter, self).__init__(model, **kwargs)
        self.enabled_parameter = enabled_parameter
        self.disabled_parameter = disabled_parameter
        self.children.add(enabled_parameter)
        self.children.add(disabled_parameter)
        self._value = value
        self.integer_size = 1

    cdef calc_values(self, Timestep timestep):
        cdef int i
        cdef int n = self.__values.shape[0]
        for i in range(n):
            if self._value == 0:
                self.__values[i] = self.disabled_parameter.__values[i]
            else:
                self.__values[i] = self.enabled_parameter.__values[i]
            self.__indices[i] = self._value

    cpdef set_integer_variables(self, int[:] values):
        self._value = values[0]

    cpdef int[:] get_integer_variables(self):
        return np.array([self._value, ], dtype=np.int32)

    cpdef int[:] get_integer_lower_bounds(self):
        return np.zeros(self.integer_size, dtype=np.int32)

    cpdef int[:] get_integer_upper_bounds(self):
        return np.ones(self.integer_size, dtype=np.int32)

    @classmethod
    def load(cls, model, data):
        enabled_parameter = load_parameter(model, data.pop("enabled_parameter"))
        disabled_parameter = load_parameter(model, data.pop("disabled_parameter"))
        return cls(model, enabled_parameter, disabled_parameter, **data)
BinaryVariableParameter.register()