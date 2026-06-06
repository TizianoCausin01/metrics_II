import os, sys, yaml
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])
from useful_stuff.general_utils.utils import TimeSeries
from useful_stuff.general_utils.II import InformationImbalance, dynInformationImbalance


def init_static_dynII(ba_raster: "TimeSeries", signal_RDM_metric, model_RDM_metric, k) -> "dynInformationImbalance":
    dyn_ii_obj = dynInformationImbalance(signal_RDM_metric, model_RDM_metric, k)
    dyn_ii_obj.compute_RDM_timeseries(ba_raster, "signal")
    dyn_ii_obj.compute_distance_ranks_timeseries("signal")
    return  dyn_ii_obj
# EOF
