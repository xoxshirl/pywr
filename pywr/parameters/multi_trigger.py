from pywr.parameters import IndexParameter, load_parameter


class MultiTriggerParameter(IndexParameter):
    """A parameter that returns the index of the first triggered sub-parameter."""
    def __init__(self, model, triggers, **kwargs):
        super().__init__(model, **kwargs)

        self.triggers = []
        for trigger in triggers:
            self.children.add(trigger)
            self.triggers.append(trigger)

    def index(self, ts, scenario_index):

        for i, trigger in enumerate(self.triggers):
            # Evaluate the triggers in order.
            value = trigger.get_index(scenario_index)
            if value > 0:
                # If triggered (index != 0) return the 1-based index of the triggers.
                index = i + 1
                break
        else:
            # If no triggers are active return 0
            index = 0
        return index

    @classmethod
    def load(cls, model, data):

        triggers = []
        for trigger_data in data.pop('triggers'):
            parameter = load_parameter(model, trigger_data)
            triggers.append(parameter)

        return cls(model, triggers, **data)

MultiTriggerParameter.register()