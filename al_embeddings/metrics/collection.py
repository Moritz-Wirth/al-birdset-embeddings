from skactiveml.utils import call_func

class MetricsCollection:
    def __init__(self, metrics_dict: dict[str, callable]):
        self.metrics_dict = metrics_dict


    def __call__(self, *args, **kwargs) -> dict[str, float]:
        scores = dict()
        if "y_proba" in kwargs:
            kwargs = {**kwargs, "y_score": kwargs["y_proba"]}
        for name, metric in self.metrics_dict.items():
            try:
                score = call_func(metric, **kwargs)
                scores[name] = score
            except Exception as e:
                for k,v in kwargs.items():
                    print(k, v.shape)
                raise Exception(f"Error calculating metric '{name}': {e}") from e

        return scores