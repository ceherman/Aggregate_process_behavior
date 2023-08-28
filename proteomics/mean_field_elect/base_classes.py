# Original work is in the Python folder of 2014 laptop files
# Order for previous log of charged aa:  pKa, z, surf_dens [mol m-2], pKa_eff, n ligands

import numpy as np
from Bio import SeqUtils


class constants:
    """Store physical constants for function access"""
    def __init__(self):
        self.e          = 1.602e-19      # [C], Fundamental charge unit
        self.Na         = 6.022e23       # [mol-1], Avogadro's number
        self.kT         = 4.11e-21       # [J], Energy conversion (for 298 K)
        self.eps0       = 8.854e-12      # [C V-1 m-1], Vacuum permittivity
        self.h          = 6.626e-34      # [J s], Planck constant
        self.F          = 96485          # [C mol-1], Faraday constant
        self.kg_per_Da  = 1.6605e-27     # [kg Da-1], Mass conversion
        self.prot_vbar  = 0.73           # [ml g-1], Protein partial specific volume
        self.hamaker    = 8.1e-21        # [J mol-1], Protein-resin Hamaker constant
        return

    def get_pKa(self, chemical):
        pKa_dic =  { # Acids
                    'C_term':      3.30, \
                    'E':           4.18, \
                    'D':           3.53, \
                    'Y':          10.26, \
                    'sulfonate':   2.30, \
                    # Bases
                    'N_term':      7.66, \
                    'K':          10.59, \
                    'R':          12.00, \
                    'H':           6.55, \
                    'quat_amine': 20.00  }
        return pKa_dic[chemical]

    def is_acid(self, chemical):
        if chemical in ['C_term', 'E', 'D', 'Y', 'sulfonate']:
            return True
        else:
            return False

    def is_base(self, chemical):
        if chemical in ['N_term', 'K', 'R', 'H', 'quat_amine']:
            return True
        else:
            return False

    def get_chemical_info(self, chemical):
        pKa             = self.get_pKa(chemical)
        is_base_flag    = self.is_base(chemical)
        is_acid_flag    = self.is_acid(chemical)
        return pKa, is_base_flag, is_acid_flag


class solution:
    """Store properties of the aqueous solution"""
    def __init__(self, pH, ion_str, eps=80.1, m_z=1):
        self.pH         = pH
        self.ion_str    = ion_str  # [M]
        self.eps        = eps      # [-]
        self.m_z        = m_z      # (Symmetrical) electorlyte valence
        self.get_ch()
        self.get_kappa()
        self.get_cap_dif()
        return

    def get_ch(self):
        """Convert pH to bulk solution hydronium ion concentration [M]"""
        self.ch = 10.0**(-1.0*self.pH)
        return

    def get_kappa(self):
        """Convert ionic strength [M] into the inverse Debye length [m-1]"""
        self.kappa = np.sqrt(2*constants().e**2*self.ion_str*constants().Na*1.0e3/\
                        (constants().kT*self.eps*constants().eps0))
        return

    def get_cap_dif(self):
        """Compute the diffuse layer capacitance [C m-2 V-1]"""
        self.cap_dif = self.eps*constants().eps0*self.kappa
        return


class resin:
    """Store resin properties"""
    def __init__(self, solution, ligand, surf_dens, name='resin_name'):
        self.name       = name
        self.ligand     = ligand
        self.surf_dens  = surf_dens # [mol m-2]
        self.solution   = solution
        self.pKa        = constants().get_pKa(self.ligand)
        self.is_acid    = constants().is_acid(self.ligand)
        self.is_base    = constants().is_base(self.ligand)
        return


class protein:
    def __init__(self, solution, seq, name='protein_name'):
        self.name       = name
        self.seq        = seq
        self.solution   = solution
        self.get_mass()
        self.get_size()
        self.get_aa_counts()
        return

    def get_mass(self):
        self.mass = SeqUtils.molecular_weight(self.seq, seq_type='protein') # [g mol-1]
        return

    def get_size(self):
        self.radius = (0.75/np.pi*self.mass*constants().prot_vbar/constants().Na)**(1/3)*1.0e-2
                                                        # [m]
        self.volume = 4.0/3.0 * np.pi * self.radius**3  # [m3]
        self.area   = 4.0 * np.pi * self.radius**2      # [m2]
        return

    def get_aa_counts(self):
        self.aa_counts = {}
        for aa in ['K', 'R', 'H', 'E', 'D', 'Y']:
            self.aa_counts[aa] = self.seq.count(aa)
        self.aa_counts['N_term'] = 1
        self.aa_counts['C_term'] = 1
        return


# Common functions______________________________________________________________
def check_bcs(bc_charge, bc_potential):
    if bc_charge==bc_potential:
        raise ValueError('Need to specify one boundary condition.')
    return

# I modified these recently
def get_Keq_nd(phi_inv_m, keq_dim):
    """Convert dimensional Keq [m] into dimensionless Keq"""
    return phi_inv_m * keq_dim

def get_kprime(total_porosity, phi_inv_m, keq_dim):
    """Convert dimensional Keq [m] into k' """
    keq_nd = get_Keq_nd(phi_inv_m, keq_dim)
    kprime = (1.0 - total_porosity)/total_porosity * keq_nd
    return kprime
