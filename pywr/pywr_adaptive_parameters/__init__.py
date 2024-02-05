""" This module contains `Parameter` subclasses for modelling transient changes.

Examples include the modelling of a decision at a fixed point during a simulation.

"""
from pywr._component import Component
from pywr.parameters._parameters import Parameter, ConstantParameter, IndexParameter, \
    load_parameter, parameter_registry
from pywr.parameters._thresholds import AbstractThresholdParameter
import numpy as np
import pandas
from ._binary_variable_parameter import BinaryVariableParameter
from .multi_trigger import MultiTriggerParameter
from .ann import ANNParameter, ANNOutputParameter


class TransientDecisionParameter(Parameter):
    """ Return one of two values depending on the current time-step

    This `Parameter` can be used to model a discrete decision event
     that happens at a given date. Prior to this date the `before`
     value is returned, and post this date the `after` value is returned.

    Parameters
    ----------
    decision_date : string or pandas.Timestamp
        The trigger date for the decision.
    before_parameter : Parameter
        The value to use before the decision date.
    after_parameter : Parameter
        The value to use after the decision date.
    earliest_date : string or pandas.Timestamp or None
        Earliest date that the variable can be set to. Defaults to `model.timestepper.start`
    latest_date : string or pandas.Timestamp or None
        Latest date that the variable can be set to. Defaults to `model.timestepper.end`
    decision_freq : pandas frequency string (default 'AS')
        The resolution of feasible dates. For example 'AS' would create feasible dates every
        year between `earliest_date` and `latest_date`. The `pandas` functions are used
        internally for delta date calculations.

    """

    def __init__(self, model, decision_date, before_parameter, after_parameter,
                 earliest_date=None, latest_date=None, decision_freq='AS', **kwargs):
        super(TransientDecisionParameter, self).__init__(model, **kwargs)
        self._decision_date = None
        self.decision_date = decision_date

        if not isinstance(before_parameter, Parameter):
            raise ValueError('The `before` value should be a Parameter instance.')
        before_parameter.parents.add(self)
        self.before_parameter = before_parameter

        if not isinstance(after_parameter, Parameter):
            raise ValueError('The `after` value should be a Parameter instance.')
        after_parameter.parents.add(self)
        self.after_parameter = after_parameter

        # These parameters are mostly used if this class is used as variable.
        self._earliest_date = None
        self.earliest_date = earliest_date

        self._latest_date = None
        self.latest_date = latest_date

        self.decision_freq = decision_freq
        self._feasible_dates = None
        self.integer_size = 1  # This parameter has a single integer variable

    def decision_date():
        def fget(self):
            return self._decision_date

        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._decision_date = value
            else:
                self._decision_date = pandas.to_datetime(value)

        return locals()

    decision_date = property(**decision_date())

    def earliest_date():
        def fget(self):
            if self._earliest_date is not None:
                return self._earliest_date
            else:
                return self.model.timestepper.start

        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._earliest_date = value
            else:
                self._earliest_date = pandas.to_datetime(value)

        return locals()

    earliest_date = property(**earliest_date())

    def latest_date():
        def fget(self):
            if self._latest_date is not None:
                return self._latest_date
            else:
                return self.model.timestepper.end

        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._latest_date = value
            else:
                self._latest_date = pandas.to_datetime(value)

        return locals()

    latest_date = property(**latest_date())

    def setup(self):
        super(TransientDecisionParameter, self).setup()

        # Now setup the feasible dates for when this object is used as a variable.
        self._feasible_dates = pandas.date_range(self.earliest_date, self.latest_date,
                                                 freq=self.decision_freq)

    def value(self, ts, scenario_index):

        if ts is None:
            v = self.before_parameter.get_value(scenario_index)
        elif ts.datetime >= self.decision_date:
            v = self.after_parameter.get_value(scenario_index)
        else:
            v = self.before_parameter.get_value(scenario_index)
        return v

    def get_integer_lower_bounds(self):
        return np.array([0, ], dtype=np.int)

    def get_integer_upper_bounds(self):
        return np.array([len(self._feasible_dates) - 1, ], dtype=np.int)

    def set_integer_variables(self, values):
        # Update the decision date with the corresponding feasible date
        self.decision_date = self._feasible_dates[values[0]]

    def get_integer_variables(self):
        return np.array([self._feasible_dates.get_loc(self.decision_date), ], dtype=np.int)

    def dump(self):

        data = {
            'earliest_date': self.earliest_date.isoformat(),
            'latest_date': self.latest_date.isoformat(),
            'decision_date': self.decision_date.isoformat(),
            'decision_frequency': self.decision_freq
        }

        return data

    @classmethod
    def load(cls, model, data):

        before_parameter = load_parameter(model, data.pop('before_parameter'))
        after_parameter = load_parameter(model, data.pop('after_parameter'))

        return cls(model, before_parameter=before_parameter, after_parameter=after_parameter, **data)
TransientDecisionParameter.register()


class ScenarioTreeDecisionItem(Component):
    def __init__(self, model, name, end_date, **kwargs):
        super(ScenarioTreeDecisionItem, self).__init__(model, name, **kwargs)
        self.end_date = end_date
        self.scenarios = []

    @property
    def start_date(self):
        # Find if there is a parent stage in the tree
        for parent in self.parents:
            if isinstance(parent, ScenarioTreeDecisionItem):
                # The start of this stage is the end of parent stage
                return parent.end_date
        # Otherwise the start is the start of the model
        return self.model.timestepper.start

    @property
    def paths(self):
        if len(self.children) == 0:
            yield (self,)
        else:
            for child in self.children:
                for path in child.paths:
                    yield tuple([self, ] + [c for c in path])

    def end_date():
        def fget(self):
            if self._latest_date is not None:
                return self._latest_date
            else:
                return self.model.timestepper.end

        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._latest_date = value
            else:
                self._latest_date = pandas.to_datetime(value)

        return locals()

    end_date = property(**end_date())


class ScenarioTreeDecisionParameter(Parameter):
    def __init__(self, model, root, scenario_selector, parameter_factory, **kwargs):
        super(ScenarioTreeDecisionParameter, self).__init__(model, **kwargs)
        self.root = root
        self.children.add(scenario_selector)
        self.scenario_selector = scenario_selector
        self.parameter_factory = parameter_factory

        self.path_index = None
        self._cached_paths = None
        self.parameters = None
        # Setup the parameters associated with the tree
        self._create_scenario_parameters()

    def _create_scenario_parameters(self):
        parameters = {}

        def make_parameter(scenario):
            p = self.parameter_factory(self.model, scenario)
            parameters[scenario] = p
            self.children.add(p)  # Ensure that these parameters are children of this
            # Recursively call to make parameters for children
            for child in scenario.children:
                make_parameter(child)

        make_parameter(self.root)
        self.parameters = parameters

    def setup(self):
        super(ScenarioTreeDecisionParameter, self).setup()
        # During setup we take the tree to scenario mapping to make
        # a more efficiency index based lookup array

        # Cache the scenario tree paths
        self._cached_paths = tuple(p for p in self.root.paths)

    def value(self, ts, scenario_index):

        i = self.scenario_selector.get_index(scenario_index)
        path = self._cached_paths[i]

        for stage in path:
            if ts.datetime < stage.end_date:
                parameter = self.parameters[stage]
                return parameter.get_value(scenario_index)

        raise ValueError('No parameter found from stages for current time-step.')

    @classmethod
    def load(cls, model, data):

        scenario_selector = load_parameter(model, data.pop('scenario_selector'))

        # Load the stages
        tree_data = data.pop('tree')
        stage_parents = {}
        stages = {}
        root_stage = None
        for stage_name, stage_data in tree_data.items():
            # Take the parent information and apply it after all stages are loaded
            parent = stage_data.pop('parent', None)
            stage = ScenarioTreeDecisionItem(model, stage_name, **stage_data)

            stages[stage_name] = stage
            if parent is not None:
                stage_parents[stage_name] = parent
            else:
                if root_stage is not None:
                    raise ValueError('Multiple stages with no parent. Only one stage can be the root stage.')
                root_stage = stage

        if root_stage is None:
            raise ValueError('No stages defined without a parent. There must be one root stage.')

        # Apply the parent-child relationship
        for stage_name, parent in stage_parents.items():
            stages[parent].children.add(stages[stage_name])

        # Create the factory function for generating the parameters in each stage of the tree
        factory_data = data.pop('parameter_factory')
        factory_type = factory_data.pop('type')

        def factory(model, stage):
            parameter_data = {k: v[stage.name] for k, v in factory_data.items()}
            parameter_data['type'] = factory_type
            return load_parameter(model, parameter_data)

        return cls(model, root_stage, scenario_selector, factory, **data)
ScenarioTreeDecisionParameter.register()


class TransientScenarioTreeDecisionParameter(ScenarioTreeDecisionParameter):
    def __init__(self, model, root, scenario_selector, enabled_parameter_factory, decision_freq='AS',
                 transient_is_variable=False, wrap_binary=False, **kwargs):

        self.enabled_parameter_factory = enabled_parameter_factory

        # These parameters are mostly used if this class is used as variable.
        self.decision_freq = decision_freq
        self.transient_is_variable = transient_is_variable
        self.wrap_binary = wrap_binary

        super(TransientScenarioTreeDecisionParameter, self).__init__(model, root, scenario_selector,
                                                                     self._transient_parameter_factory,
                                                                     **kwargs)

    def _transient_parameter_factory(self, model, stage):
        """ Private factory function for creating the transient parameters """

        name = '{}.{}.{}'.format(self.name, stage.name, '{}')

        # When the parameter is not active (either off or before decision) data
        # default to a zero value.
        # TODO make this disabled value configurable.
        disabled_parameter = ConstantParameter(model, 0, name=name.format('disabled'))

        # Use the given factory function to create the enabled parameter
        enabled_parameter = self.enabled_parameter_factory(model, stage)
        enabled_parameter.name = name.format('enabled')

        # Make the transient parameter
        earliest_date = stage.start_date
        latest_date = stage.end_date
        current_date = latest_date

        p = TransientDecisionParameter(model, current_date, disabled_parameter, enabled_parameter,
                                       earliest_date=earliest_date, latest_date=latest_date,
                                       decision_freq=self.decision_freq, is_variable=self.transient_is_variable,
                                       name=name.format('transient'))

        # Finally wrap the transient parameter in a binary variable if required.
        if not self.wrap_binary:
            return p
        else:
            return BinaryVariableParameter(model, p, disabled_parameter, is_variable=self.transient_is_variable,
                                           name=name.format('binary'))

    def value(self, ts, scenario_index):
        i = self.scenario_selector.get_index(scenario_index)
        path = self._cached_paths[i]

        value = 0.0
        for stage in path:
            parameter = self.parameters[stage]

            # The stages are iterated through in time order (first to last)
            # Therefore if this stage has an active binary variable we
            # ignore the current time-step and use the value of this stage's parameter
            if ts.datetime < stage.end_date:
                value += parameter.get_value(scenario_index)

        return value
TransientScenarioTreeDecisionParameter.register()


class NormalisedParameter(Parameter):
    def __init__(self, model, parameter, **kwargs):
        self.max = kwargs.pop('max')
        self.min = kwargs.pop('min')

        super().__init__(model, **kwargs)
        self.children.add(parameter)
        self.parameter = parameter

    def value(self, ts, si):

        value = self.parameter.get_value(si)
        return (value - self.min)/(self.max - self.min)

    @classmethod
    def load(cls, model, data):
        parameter = load_parameter(model, data.pop('parameter'))
        return cls(model, parameter, **data)
NormalisedParameter.register()


class DelayIndexParameter(IndexParameter):
    def __init__(self, model, parameter, delay, **kwargs):
        super().__init__(model, kwargs)
        self.parameter = parameter
        self.children.add(parameter)
        self.delay = delay
        self._memory = None

    def reset(self):
        # Create empty memory for each scenario
        ncomb = len(self.model.scenarios.combinations)
        self._memory = [[] for _ in range(ncomb)]

    def index(self, ts, si):

        current_index = self.parameter.get_index(si)
        if self.delay == 0:
            i = current_index
        else:
            memory = self._memory[si.global_id]
            if len(memory) >= self.delay:
                i = memory.pop(0)
            else:
                i = 0
            memory.append(current_index)
        return i
    @classmethod
    def load(cls, model, data):
        parameter = load_parameter(model, data.pop('parameter'))
        return cls(model, parameter, **data)
DelayIndexParameter.register()


class DatetimeThresholdParameter(AbstractThresholdParameter):
    """ Returns one of two values depending on the value of a Parameter

    Parameters
    ----------
    recorder : `pywr.core.AbstractNode`

    """
    def _value_to_compare(self, timestep, scenario_index):
        dt = timestep.datetime
        dt64 = np.datetime64(dt).astype('datetime64[D]')
        return dt64.astype(float)

    @classmethod
    def load(cls, model, data):
        threshold = load_parameter(model, data.pop("threshold"))
        values = data.pop("values", None)
        predicate = data.pop("predicate", None)
        return cls(model, threshold, values=values, predicate=predicate, **data)
DatetimeThresholdParameter.register()


class DatetimeParameter(ConstantParameter):
    def __init__(self, model, value, lower_bounds, upper_bounds, **kwargs):
        value = np.datetime64(value).astype('datetime64[D]').astype(float)
        lower_bounds = np.datetime64(lower_bounds).astype('datetime64[D]').astype(float)
        upper_bounds = np.datetime64(upper_bounds).astype('datetime64[D]').astype(float)
        super().__init__(model, value, lower_bounds=lower_bounds,
                         upper_bounds=upper_bounds, **kwargs)
DatetimeParameter.register()
