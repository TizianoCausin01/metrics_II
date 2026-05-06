
"""
decode_matlab_strings
Decodes MATLAB strings stored in a v7.3 .mat file (HDF5 format) into Python strings.
1) Iterates over HDF5 object references pointing to MATLAB char arrays
2) Reads the corresponding uint16 character codes
3) Converts character codes to Python characters and joins them into strings

INPUT:
- h5file: h5py.File -> open HDF5 file corresponding to a MATLAB v7.3 .mat file
- ref_array: np.ndarray -> array of HDF5 object references to MATLAB char arrays

OUTPUT:
- strings: list of str -> decoded MATLAB strings
"""
def decode_matlab_strings(h5file, ref_array):
    strings = []
    for ref in ref_array.squeeze():
        chars = h5file[ref][:]
        s = ''.join(chr(c) for c in chars.flatten()) # MATLAB chars are usually stored as Nx1 uint16
        strings.append(s)
    return strings

"""
load_img_natraster
Loads and preprocesses natural image raster data for a given monkey/session.

1) Loads the MATLAB v7.3 natraster file (HDF5 format)
2) Casts data to float32 and reorders axes to (neurons, time, trials)
3) Optionally slices the signal to a specific brain area
4) Wraps the data in a TimeSeries object
5) Resamples the signal to the target sampling frequency

INPUT:
- paths: dict[str, str] -> dictionary containing base data paths
- cfg: Cfg -> configuration object with required attributes:
    * monkey_name: str
    * date: str
    * new_fs: float
    * brain_area (optional): str

OUTPUT:
- rasters: TimeSeries -> preprocessed neural raster time series
"""
def load_img_natraster(paths: dict[str: str], monkey_name, date, new_fs=None, brain_area=None):
    rasters_path = f"{paths['livingstone_lab']}/tiziano/data/{monkey_name}_natraster{date}.mat"
    with h5py.File(rasters_path, "r") as f:
        rasters = f["natraster"][:]      
    rasters = rasters.astype(np.float32)
    rasters = rasters.transpose(2, 1, 0)
    rasters = TimeSeries(rasters, 1000)
    if brain_area is not None:
            brain_areas_obj = BrainAreas(monkey_name)
            rasters = brain_areas_obj.slice_brain_area(rasters, brain_area)
    # end if brain_area is not None:
    if new_fs is not None:
        rasters.resample(new_fs)
    # if new_fs is not None:
    return rasters
# EOF

"""
BrainAreas
Utility class for slicing neural data into predefined brain areas.
1) Loads brain-area channel indices from a YAML configuration file
2) Validates input rasters against the expected number of channels
3) Extracts and concatenates channel ranges corresponding to a given brain area

INPUT:
- monkey_name: str -> identifier used to select the correct brain-area mapping

OUTPUT (slice_brain_area):
- brain_area_response: np.ndarray -> subset of rasters corresponding to the selected brain area
"""
class BrainAreas:
    def __init__(self, monkey_name: str):
        self.monkey_name = monkey_name
        with open("../../brain_areas.yaml", "r") as f:
            config = yaml.safe_load(f)
        try:
            self.areas_idx = config[self.monkey_name]
            self.brain_areas = [k for k in self.areas_idx.keys() if k!='n_chan']
        except KeyError:
            raise KeyError(f"Monkey '{self.monkey_name}' not found.", f"Supported monkeys {list(config.keys())}") from None
        # end try:
    # EOF
    # --- GETTERS ---
    def get_brain_areas_idx(self):
        return self.areas_idx
    #EOF
    def get_brain_areas(self):
        return self.brain_areas
    #EOF
    # --- OTHER FUNCTIONS ---
    def slice_brain_area(self, rasters: "TimeSeries", brain_area_name: str):
        if rasters.get_array().shape[0] < self.areas_idx["n_chan"][0]:
            raise ValueError(f"Rasters of shape {rasters.get_array().shape} doesn't match the original number of channels ({self.areas_idx["n_chan"]}).")
        # end if rasters.shape[0] < self.areas_idx["n_chan"][0]:
        try:
            target_brain_area = self.areas_idx[brain_area_name]
        except KeyError:
            raise KeyError(f"Brain area '{brain_area_name}' not found for monkey '{self.monkey_name}'.", f"Supported brain areas: {list(self.areas_idx.keys())}") from None
            
        except TypeError:
            if isinstance(brain_area_name, list) and len(brain_area_name) == 2:
                for idx in brain_area_name:
                    if idx > self.areas_idx["n_chan"][0]:
                        raise ValueError(f"Indices passed {brain_area_name} don't match the original number of channels ({self.areas_idx["n_chan"]}).")
                    # end if idx > self.areas_idx["n_chan"][0]:
                # end for idx in brain_area_name:
                target_brain_area = [brain_area_name] # it's setting the limits in terms of channels idx where we don't have precise info about a brain area, wrapping them in a list of lists
            else:
                raise TypeError(f"brain_area_name should be either a str or a list of len 2.")
            # end if isinstance(brain_area_name, list) and len(brain_area_name) == 2:
        # end try:
        brain_area_response = []
        for lims in target_brain_area:
            start, end = lims
            brain_area_response.append(rasters.get_array()[start:end, ...])
        # end for lims in target_brain_area:
        brain_area_response = np.concatenate(brain_area_response)
        brain_area_response = TimeSeries(brain_area_response, rasters.fs)
        return brain_area_response
    # EOF
# EOC

