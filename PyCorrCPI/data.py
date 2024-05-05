from functools import wraps, partial
from typing import Union, Optional
import json
import os

from .imports import *
from .helpers_numba import *
from .covariance import *
from .helpers_tof import TofCalibration, get_ion_mass, get_ion_charge, get_ion_mz, parse_chemical_formula



class Ion:
    """Class used to define the parameters to extract data from different ions from a dataset"""
    def __init__(self, label, filter_i, filter_f, dataset=None, filter_param='tof', filter_param_type:str="abs",
                center_x=None, center_y=None, center_t=None, mass=None, charge=None, shot_array_method='range', allow_auto_mass_charge=False, fit_t=False,
                C_xy=None, C_z=None, filter_query:str=None, mz_calibration:TofCalibration=None
                ):
        """
        Parameters
            label (str):
            filter_i (float): start of tof gate
            filter_f (float): end of tof gate
            dataset (DataSet, optional):
            filter_param (str, optional):
            filter_param_type (str, optional):
                specifies how to interpret the filter range provided (except `abs`, filter_param=`tof` required)
                - `abs`: gives an absolute lower and upper bound for the given filter_param
                - `rel`: gives a range relative to center_t or fitted_center_t in the units of tof (depends on fit_t)
                - `frac`: gives a range as a fraction of center_t or fitted_center_t (depends on fit_t)
            center_x (float, optional):
            center_y (float, optional):
            center_t (float, optional):
            mass (float, optional):
            charge (float, optional):
            shot_array_method (str, optional):
            allow_auto_mass_charge (bool, optional):
            fit_t (bool, optional): if true, when a calibration available use fitted t_center
            C_xy (float, optional):
            C_z (float, optional):
            filter_query (str, optional):
                query that will be applied to the ions data frame, 
                enables realization of comples rois to exclude data (e.g. warm background)
            mz_calibration (TofCalibration, optional):
                ToF calibration to have a fitted fitted_center_t
        
        """
        self._data_array = None
        
        self.label = label
        self._filter_i = filter_i
        self._filter_f = filter_f
        self.filter_query = filter_query
        self.filter_param = filter_param
        self.filter_param_type = filter_param_type
        self.shot_array_method = shot_array_method
        self.allow_auto_mass_charge = allow_auto_mass_charge
        self.mz_calibration = mz_calibration
        self.fit_t = fit_t
        
        self.C_xy = C_xy # for momentum calib
        self.C_z = C_z # for momentum calib
        self.cal_mom = False
        
        if not self.allow_auto_mass_charge or mass:
            self.mass = mass
            self.auto_mass = False
        else:
            try:
                self.mass = get_ion_mass(label)
                self.auto_mass = True
            except Exception as e:
                print(f"Could not auto-compute ion mass for {label}. Error: {e}")
                self.auto_mass = False
                self.mass = None
        
        if not self.allow_auto_mass_charge or charge:
            self.charge = charge
            self.auto_charge = False
        else:
            try:
                self.charge = get_ion_charge(label)
                self.auto_charge = True
            except Exception as e:
                print(f"Could not auto-compute ion charge for {label}. Error: {e}")
                self.auto_charge = False
                self.charge = None

        if center_x:
            self.center_x=center_x
        else:
            self.center_x=0
        if center_y:
            self.center_y=center_y
        else:
            self.center_y=0
        
        if not [i for i in (center_x,center_y) if i is None]:
            self.center_given=True
        else:
            self.center_given=False
        self._center_t = center_t
        if dataset:
            self.assign_dataset(dataset)
            
    def set_filter_gates(self, filter_i, filter_f, filter_param=None, filter_param_type=None):
        if filter_param:
            self.filter_param=filter_param
        if filter_param_type:
            self.filter_param_type=filter_param_type
        self._filter_i = filter_i
        self._filter_f = filter_f
        
        self.reset_data()
        
    
    @property
    def filter_i(self):
        if self.filter_param_type == 'abs':
            return self._filter_i
        elif self.filter_param_type == 'rel':
            return self.center_t+self._filter_i
        elif self.filter_param_type == 'frac':
            return self.center_t * self._filter_i
        
        raise ValueError("Cannot compute `filter_i` invalid filter_param_type config")
    
    @property
    def filter_f(self):
        if self.filter_param_type == 'abs':
            return self._filter_f
        elif self.filter_param_type == 'rel':
            return self.center_t+self._filter_f
        elif self.filter_param_type == 'frac':
            return self.center_t * self._filter_f
        
        raise ValueError("Cannot compute `filter_f` invalid filter_param_type config")
    
    @property
    def config(self):
        return dict(
            label=self.label, 
            filter_i=self._filter_i, 
            filter_f=self._filter_f, 
            filter_param=self.filter_param,
            filter_param_type=self.filter_param_type,
            center_x=self.center_x, 
            center_y=self.center_y, 
            center_t=self._center_t, 
            mass=self.mass, 
            charge=self.charge, 
            shot_array_method=self.shot_array_method, 
            allow_auto_mass_charge=self.allow_auto_mass_charge,
            fit_t = self.fit_t,
            C_xy=self.C_xy,
            C_z=self.C_z,
            filter_query=self.filter_query,
            mz_calibration=self.mz_calibration.coeffs,
        )

            
    def print_details(self):
        for key,value in self.__dict__.items():
            if key not in ['data_df','data_array', 'shot_array']:
                print(f"'{key}':{value}")
    
    def reset_data(self):
        self.data_df = None
        self.shot_array = None
        self.reset_data_array()
    
    def grab_data(self,dataset):
        """Gets data corresponding to ion from dataset based on the inputted filter"""
        try:
            self.data_df = dataset.sep_by_custom(self.filter_i,self.filter_f, self.filter_param)
        except:
            raise Exception("filter_param is not found in dataframe!")
        if self.filter_query:
            center_x = self.center_x
            center_y = self.center_y
            center_t = self.center_t
            self.data_df = self.data_df.query(self.filter_query)
        self.reset_data_array()
        
    @property
    def data_array(self):
        if self._data_array is None:
            self._data_array = self.data_df[["px_AU", "py_AU", "pz_AU", "shot", "pmag_AU"]].to_numpy()
        return self._data_array

    def reset_data_array(self):
        """Converts necessary dataframe columns for covariance calculation to an array"""
        self._data_array = None
        self.cal_mom = False
    
    def assign_dataset(self, dataset):
        self.grab_data(dataset)
        self.get_shot_array()

    def get_shot_array(self):
        """Find array of shots in dataset which contain this ion"""
        if len(self.data_df) == 0:
            self.shot_array = np.array([])
            return 
        if self.shot_array_method=='range':
            self.shot_array = np.arange(np.min(self.data_df.shot), np.max(self.data_df.shot))
        elif self.shot_array_method=='unique':
            self.shot_array = np.array(np.unique(self.data_df.shot))
        else:
            raise Exception("Invalid shot_array_method inputted!")

    def get_idx_dict(self, shot_array_total):
        """Create dictionary of indices of rows in dataset corresponding to this ion. Needed for covariance"""
        idx_dict = Dict.empty(
                key_type=float_single,
                value_type=float_array)
        
        calculate_indexes(idx_dict,self.shot_array,shot_array_total,self.data_array)
        self.idx_dict=idx_dict
    
    @property
    def mz(self):
        return self.mass/self.charge
        
    @property
    def fitted_center_t(self):
        if self.mz_calibration.coeffs:
            return self.mz_calibration.mz_to_tof(self.mz)
        
    @property
    def center_t(self):
        if self.fit_t and self.mz_calibration.coeffs:
            return self.fitted_center_t
        return self._center_t

    def calibrate_momenta(self, t0:float, C_xy=None, C_z=None, fit_center=False):
        data_df_ion = self.data_df
        
        if C_xy:
            self.C_xy = C_xy
        if C_z:
            self.C_z = C_z
            
        if not self.C_xy or not self.C_z:
            raise ValueError("C_xy and C_z need to be specified")
        
        if fit_center:
            # fit center not implemented yet
            data_df_ion['x_centered'] = data_df_ion-self.centre_x_fit
            data_df_ion['y_centered'] = data_df_ion-self.centre_y_fit
        else:
            data_df_ion['x_centered'] = data_df_ion.x-self.center_x
            data_df_ion['y_centered'] = data_df_ion.y-self.center_y
            
        center_t = self.center_t
            
        # if this is the first time the dataset has had momentum calibration run on it, populate new columns
        # in the dataframe
        if self.cal_mom==False:
            data_df_ion['ion'] = np.nan
            data_df_ion['t_relative'] = np.nan
            data_df_ion['vx'] = np.nan
            data_df_ion['vy'] = np.nan
            data_df_ion['vz'] = np.nan
            data_df_ion['px'] = np.nan
            data_df_ion['py'] = np.nan
            data_df_ion['pz'] = np.nan
            data_df_ion['vmag'] = np.nan
            data_df_ion['pmag'] = np.nan
            self.cal_mom=True
        
        data_df_ion['t_absolute'] = data_df_ion['tof']-t0
        data_df_ion['t_centered'] = data_df_ion['tof']-center_t
        
        data_df_ion['vx'] = self.C_xy*(data_df_ion['x_centered'])#/data_df_ion['tcorr'])
        data_df_ion['vy'] = self.C_xy*(data_df_ion['y_centered'])#/data_df_ion['tcorr'])
        data_df_ion['px'] = data_df_ion['vx'] * self.mass
        data_df_ion['py'] = data_df_ion['vy'] * self.mass
        data_df_ion['vz'] = (self.C_z*self.charge*(data_df_ion['t_centered']))/self.mass
        data_df_ion['pz'] = data_df_ion['vz'] * self.mass
        
        data_df_ion['pmag'] = np.sqrt((data_df_ion['px']**2+data_df_ion['py']**2+data_df_ion['pz']**2))
        data_df_ion['vmag'] = np.sqrt((data_df_ion['vx']**2+data_df_ion['vy']**2+data_df_ion['vz']**2))
        
        return(data_df_ion)    


class IonCollection:
    """Class used to define/manage a set of ions"""
    def __init__(
        self, center_x:Optional[float]=None, 
        center_y:Optional[float]=None, 
        filter_param:Optional[str]=None, 
        allow_auto_mass_charge:Optional[bool]=None, 
        fit_t:Optional[bool]=None, 
        shot_array_method:Optional[str]=None,
        C_xy:Optional[float]=None,
        C_z:Optional[float]=None,
        mz_calibration_coeffs:Optional[tuple]=None,
        allow_auto_ion_creation=False
    ):
        """
        center_x:Optional[float]=None, 
        center_y:Optional[float]=None, 
        filter_param:Optional[str]=None, 
        allow_auto_mass_charge:Optional[bool]=None, 
        fit_t:Optional[bool]=None, 
        shot_array_method:Optional[str]=None,
        C_xy:Optional[float]=None,
        C_z:Optional[float]=None,
        mz_calibration_coeffs:Optional[tuple]=None,
        allow_auto_ion_creation=False
            if true, when requesting a label that is not in the label list,
            the ion parameters are guessed
        """
        self._data = list()
        self._filter_param = filter_param
        self._allow_auto_mass_charge = allow_auto_mass_charge
        self._fit_t = fit_t
        self._shot_array_method = shot_array_method
        self._center_x = center_x
        self._center_y = center_y
        self._C_xy = C_xy
        self._C_z = C_z
        self._dataset = None
        self.allow_auto_ion_creation = allow_auto_ion_creation
        
        self.tof_mz_cal = TofCalibration(mz_calibration_coeffs)
        
        

        optional_kwargs = dict()
        if self._filter_param:
            optional_kwargs["filter_param"] = self._filter_param
        if self._allow_auto_mass_charge is not None:
            optional_kwargs["allow_auto_mass_charge"] = self._allow_auto_mass_charge
        if self._fit_t is not None:
            optional_kwargs["fit_t"] = self._fit_t
        if self._shot_array_method:
            optional_kwargs["shot_array_method"] = self._shot_array_method
        if self._center_x:
            optional_kwargs["center_x"] = self._center_x
        if self._center_y:
            optional_kwargs["center_y"] = self._center_y
        if self._C_xy:
            optional_kwargs["C_xy"] = self._C_xy
        if self._C_z:
            optional_kwargs["C_z"] = self._C_z
            
        optional_kwargs["mz_calibration"] = self.tof_mz_cal

        self._ion_class = partial(Ion, **optional_kwargs)
        
    def __getitem__(self, index):
        if isinstance(index, str):
            _labels = self.labels
            if index in _labels:
                return self._data[_labels.index(index)]
            else:
                if self.allow_auto_ion_creation:
                    try:
                        print(f"{index} not in available labels, failed to automatically add it as a new ion")
                        parse_chemical_formula(index)
                        ion = self.add_ion(index, 0.99, 1.01, filter_param_type="frac")
                        if self._dataset:
                            ion.assign_dataset(self._dataset)
                            if self.tof_mz_cal.coeffs:
                                ion.calibrate_momenta(self.tof_mz_cal.t0)
                    except:
                        raise KeyError(f"{index} not in available labels, failed to automatically add it as a new ion")
                else:
                    raise KeyError(f"{index} not in available labels")
            
        return self._data[index]
    
    def __iter__(self, *args, **kwargs):
        return self._data.__iter__(*args, **kwargs)
    
    def __len__(self):
        return len(self._data)
        
    def __str__(self):
        return str([ion.label for ion in self._data])
    
    def __repr__(self):
        return f"Collection with {len(self._data)} ions:\n{str(self)}"
    
    @property
    def data(self):
        return self._data
    
    @property
    def labels(self):
        return [ion.label for ion in self.data]
    
    @property
    def config(self):
        return dict(
            center_x=self._center_x, 
            center_y=self._center_y, 
            filter_param=self._filter_param, 
            allow_auto_mass_charge=self._allow_auto_mass_charge, 
            fit_t=self._fit_t,
            shot_array_method=self._shot_array_method,
            C_xy=self._C_xy,
            C_z=self._C_z,
            mz_calibration_coeffs=self.tof_mz_cal.coeffs,
        )
    
    def export_config_file(self, output_path:str):
        """
        Export IonCollection incl. ions into a config file
        """
        full_config = dict()
        full_config["IonCollection"] = self.config
        full_config["Ions"] = [ion.config for ion in self.data]
        if not output_path.endswith(".json"):
            output_path = f"{output_path}.json"
        with open(output_path, 'w') as f:
            json.dump(full_config, f)    
    
    @classmethod
    def from_config_file(cls, input_path, allow_auto_ion_creation=False):
        """
        Construct IonCollection incl. ions from a given config file
        usage, e.g.: cpi.IonCollection.from_config_file("test.json")
        """
        with open(input_path, 'r') as f:
            full_config = json.load(f)  
            full_config["allow_auto_ion_creation"]=allow_auto_ion_creation
        ic = cls(**full_config["IonCollection"])
        for ion_conf in full_config["Ions"]:
            if "mz_calibration" in ion_conf:
                if ion_conf["mz_calibration"] == ic.tof_mz_cal.coeffs:
                    ion_conf["mz_calibration"] = ic.tof_mz_cal
                else:
                    ion_conf["mz_calibration"] = TofCalibration(ion_conf["mz_calibration"])
            ic.add_ion(**ion_conf)
        return ic
    
    @wraps(Ion)
    def add_ion(self, *args, **kwargs):
        # create ion and append it
        ion = self._ion_class(*args, **kwargs)
        self._data.append(ion)
        return ion
        
    def assign_dataset(self, dataset):
        self._dataset = dataset
        for ion in self._data:
            ion.assign_dataset(dataset)
            
    def mz_calibration(self):
        tof_list = []
        mz_list = []
        for ion in self.data:
            if ion.mass is not None and ion.charge is not None and ion._center_t is not None:
                tof_list.append(ion.center_t)
                mz_list.append(ion.mass/ion.charge)

        mz_arr = np.array(mz_list)
        tof_arr = np.array(tof_list)
        
        coeffs, pcov = self.tof_mz_cal.calibrate(tof_arr, mz_arr)
        
        print(coeffs, pcov)
        
    def calibrate_momenta(self):
        for ion in self.data:
            ion.calibrate_momenta(self.tof_mz_cal.t0)
    

class Dataset:
    def __init__(self, data_df, C_xy = None, C_z = None, shot_array_method = 'range'):
        # print(data_df)
        self.data_df = data_df
        self.columns = list(self.data_df.columns)
        self.cal_mom = False
        self.C_xy = C_xy
        self.C_z = C_z
        self.shot_array_method = shot_array_method

        
        # assert 'x' in self.columns, "Input dataframe is missing 'x'"
        # assert 'y' in self.columns, "Input dataframe is missing 'y'"
        # assert 't' in self.columns, "Input dataframe is missing 't'"
        assert 'shot' in self.columns, "Input dataframe is missing 'shot'"

        self.get_shot_array()
    
    def __repr__(self):
        return "PyCorrCPI.Dataset with following df:\n"+repr(self.data_df)
        
        
    # def sep_by_tof(self, Ti, Tf):
    #     data_df_filt = self.data_df[(self.data_df['t']>=Ti)&(self.data_df['t']<Tf)].copy()
    #     return(data_df_filt)

    def sep_by_tof(self, Ti, Tf):
        data_df_filt = self.data_df[(self.data_df['tof']>=Ti)&(self.data_df['tof']<Tf)].copy()
        return(data_df_filt)

    def sep_by_custom(self, lim1, lim2, param):
        data_df_filt = self.data_df[(self.data_df[param]>=lim1)&(self.data_df[param]<lim2)].copy()
        return(data_df_filt)
    
    def calibrate_all_momenta(self, ion_list):
        for ion in ion_list:
            self.calibrate_momenta(ion)

    def get_shot_array(self):
        if self.shot_array_method=='range':
            self.shot_array = np.arange(np.min(self.data_df.shot), np.max(self.data_df.shot))
        elif self.shot_array_method=='unique':
            self.shot_array = np.array(np.unique(self.data_df.shot))
        else:
            raise Exception("Invalid shot_array_method inputted!")
        
        
    #### need to move this function into ion class...
    def calibrate_momenta(self, ion, C_xy=None, C_z=None, fit_center=False):
        ion_mask = self.data_df[(self.data_df['t']>=ion.Ti)&(self.data_df['t']<ion.Tf)]
        data_df_ion = self.data_df[ion_mask]
        
        if C_xy:
            self.C_xy = C_xy
        if C_z:
            self.C_z = C_z
        
        if fit_center:
            data_df_ion['x_centered'] = data_df_ion-ion.centre_x_fit
            data_df_ion['y_centered'] = data_df_ion-ion.centre_y_fit
        else:
            data_df_ion['x_centered'] = data_df_ion-ion.centre_x
            data_df_ion['y_centered'] = data_df_ion-ion.centre_y
            
        # if this is the first time the dataset has had momentum calibration run on it, populate new columns
        # in the dataframe
        if self.cal_mom==False:
            self.data_df_ion['ion'] = np.nan
            self.data_df_ion['t_relative'] = np.nan
            self.data_df_ion['vx'] = np.nan
            self.data_df_ion['vy'] = np.nan
            self.data_df_ion['vz'] = np.nan
            self.data_df_ion['px'] = np.nan
            self.data_df_ion['py'] = np.nan
            self.data_df_ion['pz'] = np.nan
            self.data_df_ion['vmag'] = np.nan
            self.data_df_ion['pmag'] = np.nan
            self.cal_mom=True
        
        self.data_df['t_absolute'] = self.data_df_ion['t']-self.t0
        self.data_df.loc[ion_mask, 't_relative'] = self.data_df.loc[ion_mask, 't']-ion.centre_t
        
        self.data_df.loc[ion_mask,'vx'] = C_xy*(self.data_df.loc[ion_mask,'x_centered']/self.data_df.loc[ion_mask,'tcorr'])
        self.data_df.loc[ion_mask,'vy'] = C_xy*(self.data_df.loc[ion_mask,'y_centered']/self.data_df.loc[ion_mask,'tcorr'])
        self.data_df.loc[ion_mask,'px'] = self.data_df.loc[ion_mask,'vx'] * ion.mass
        self.data_df.loc[ion_mask,'py'] = self.data_df.loc[ion_mask,'vy'] * ion.mass
        self.data_df.loc[ion_mask,'vz'] = (C_z*charge*(self.data_df.loc[ion_mask,'t_centered']))/ion.mass
        self.data_df.loc[ion_mask,'pz'] = self.data_df.loc[ion_mask,'vz'] * ion.mass
        
        self.data_df.loc[ion_mask,'pmag'] = np.sqrt((self.data_df.loc[ion_mask,'px']**2+self.data_df.loc[ion_mask,'py']**2+self.data_df.loc[ion_mask,'pz']**2))
        self.data_df.loc[ion_mask,'vmag'] = np.sqrt((self.data_df.loc[ion_mask,'vx']**2+self.data_df.loc[ion_mask,'vy']**2+self.data_df.loc[ion_mask,'self.data_df.loc[ion_mask,vz']**2))
        
        return(data_df_ion)
    
    ### should be moved out of class
#     def mz_calibration(self, ion_list):
#         tof_list = []
#         mz_list = []
#         for ion in ion_list:
#             if (ion.mass&ion.charge):
#                 tof_list.append(ion.centre_t)
#                 mz_list.append(ion.mass/ion.charge)

#         mz_arr = np.array(mz_list)
#         tof_arr = np.array(tof_list)

#         # Using np.polyfit to do the linear fitting
#         coeffs_sqmz_tof = np.polyfit(np.sqrt(mz_arr[mz_arr>0]), tof_arr[mz_arr>0], 1)

#         # Using IPython.Display to print LaTeX
#         display(Math(r"t = %.2f\sqrt{\frac{m}{z}} + %.2f" % (coeffs_sqmz_tof[0], coeffs_sqmz_tof[1])))

#         coeffs_tof_sqmz = np.polyfit(tof_arr[z_arr>0], np.sqrt(z_arr[z_arr>0]),1)

#         display(Math(r"\sqrt{\frac{m}{z}} = %.4ft + %.4f" % (coeffs_tof_sqmz[0], coeffs_tof_sqmz[1])))
        
#         self.t0 = coeffs_sqmz_tof[1]
        
#         self.coeffs_sqmz_tof = coeffs_sqmz_tof
#         self.coeffs_tof_sqmz = coeffs_tof_sqmz
        
#         self.apply_mz_calibration()

#         return(coeffs_sqmz_tof, coeffs_tof_sqmz)
    
    # def apply_mz_calibration(self):
    #     self.data_df['cal_mz'] = self.data_df['t']*self.coeffs_tof_sqmz[0] + self.coeffs_tof_sqmz[1]
        
    def apply_jet_correction(self):
        self.data_df['xcorr'] = self.data_df['x']
        self.data_df['ycorr'] = self.data_df['y']