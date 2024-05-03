import periodictable as pse
from scipy.optimize import curve_fit
from collections.abc import Iterable


def parse_chemical_formula(chemical_formula:str)->dict:
    """
    converts chemical formular (eg. CH2IBr) into dictionary
    with elements as key and number of atoms per element as value
    
    Parameters:
    chemical_formula (str): chemical formula with capitalized element symbols
                            followed by a number giving the count or no number
                            in case of a count of 1
    
    Returns:
    (dict): keys are the element symbols and value their count
    """
    elements = list()
    counts = list()
    for c in chemical_formula:
        if c.isupper():
            if len(elements)>len(counts):
                counts.append(1)
            elements.append(c)
        elif c.islower():
            elements[-1]+=c
        elif c.isnumeric():
            if len(elements)>len(counts):
                counts.append(c)
            else:
                counts[-1]+=c
        else:
            raise ValueError("Invalid Input")
            
    if len(elements)>len(counts):
        counts.append(1)
        
    counts = [int(c) for c in counts]
    return dict(zip(elements, counts))


def get_molecule_mass(chemical_formula:str):
    """
    gives mass in amu for any given chemical formula
    """
    
    count_by_element = parse_chemical_formula(chemical_formula)
    return sum([getattr(pse, e).mass*count_by_element[e] for e in count_by_element])


# function to parse ion strings -> i.e. [chemical formula]^[n]+
get_ion_mass = lambda ion: get_molecule_mass(ion.split("^")[0])
get_ion_charge = lambda ion: int(ion.split("^")[1][:-1])
get_ion_mz = lambda ion: get_ion_mass(ion)/get_ion_charge(ion)


class TofCalibration:
    def __init__(self, coefficients=None):
        self._coeffs = None
        
        if coefficients:
            self.coeffs = coefficients
        
    @staticmethod
    def _tof_to_mz(t, a, b):
        return ((t - a) / b) ** 2
    
    @staticmethod
    def _mz_to_tof(mz, a, b):
        return np.sqrt(mz) * b + a
    
    @property
    def coeffs(self):
        return self._coeffs
        
    @coeffs.setter
    def coeffs(self, value):
        if hasattr(value, "__len__"):
            if len(value) == 2:
                self._coeffs = (float(value[0]), float(value[1]))
                return 
        raise ValueError(f"{value} is not valid for coeffs")
        
    @property
    def t0(self):
        if self.coeffs:
            return self.coeffs[0]
        return np.nan
    
    def tof_to_mz(self, t):
        if not self.coeffs:
            raise ValueError("calibration lacks coefficients (self.coeffs is None)")
        return self._tof_to_mz(t, *self.coeffs)
    
    def mz_to_tof(self, mz):
        if not self.coeffs:
            raise ValueError("calibration lacks coefficients (self.coeffs is None)")
        return self._mz_to_tof(mz, *self.coeffs)
    
    def calibrate(self, tofs:Iterable, mzs:Iterable):
        popt, pcov = curve_fit( self._tof_to_mz, 
                                tofs, 
                                mzs)
        
        print(popt)
        self.coeffs = popt
        return self.coeffs, pcov