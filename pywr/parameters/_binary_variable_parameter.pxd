from pywr.parameters._parameters cimport Parameter, IndexParameter

cdef class BinaryVariableParameter(IndexParameter):
    cdef public Parameter enabled_parameter
    cdef public Parameter disabled_parameter
    cdef bint _value