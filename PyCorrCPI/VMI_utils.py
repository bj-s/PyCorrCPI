import periodictable as pse


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