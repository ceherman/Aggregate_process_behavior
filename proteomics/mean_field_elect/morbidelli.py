import base_classes
from base_classes import constants as con

import numpy as np
from scipy import optimize


# Classes_______________________________________________________________________

class resin(base_classes.resin):
    """Resin in Morbidelli model"""
    def __init__(self, solution, ligand, surf_dens, dpr=2e-10, name='resin_name'):
        # Note to self:  constructors aren't generally needed for subclasses
        super().__init__(solution, ligand, surf_dens, name)
        self.dpr        = dpr       # [m], protein-resin minimum resting distance
        self.pKa_eff    = pKa_eff(self.solution.ion_str, self.pKa, self.is_base, self.is_acid)
        self.get_charge_dens()
        self.get_charge_dens_dl()
        self.get_theta_0()
        self.capital_theta = capital_theta(self)
        return

    def get_charge_dens(self):
        term = get_titration_term(self.solution.ch, self.pKa_eff,
                                                self.is_base, self.is_acid)
        self.charge_dens = term*con().F*self.surf_dens  # [C m-2]
        return

    def get_charge_dens_dl(self):
        self.charge_dens_dl = self.solution.m_z*con().e*self.charge_dens/\
                            (con().kT*con().eps0*self.solution.eps*self.solution.kappa)
        return

    def get_theta_0(self):
        def residual(theta_r0):
            lhs = self.charge_dens_dl
            rhs = 4.0*np.tanh(theta_r0/4.0)/(1.0-(np.tanh(theta_r0/4.0))**2)
            return lhs - rhs

        # fit = optimize.minimize(residual, x0=[0], bounds=[(-1000, 1000)], tol=1e-12)
        fit = optimize.least_squares(residual, x0=[1.0], bounds=(-100, 100), ftol=1e-12)
        assert fit.success
        assert fit.fun[0] < 1e-6
        self.theta_0 = fit.x[0]
        return


class protein(base_classes.protein):
    """Protein in Morbidelli model"""
    def __init__(self, solution, seq, dpr=2.0e-10, name='protein_name', negative_sign=True):
        super().__init__(solution, seq, name)
        self.dpr            = dpr           # [m], protein-resin minimum resting distance
        self.negative_sign  = negative_sign # Should be true for the correct formulation
        self.get_pKa_eff()
        self.get_net_charge()
        self.get_charge_dens()
        self.get_charge_dens_dl()
        self.get_theta_0()
        self.capital_theta = capital_theta(self)
        self.get_fp()
        return

    def get_pKa_eff(self):
        self.aa_pKa_eff = {}
        for aa in self.aa_counts.keys():
            pKa, is_base, is_acid = con().get_chemical_info(aa)
            self.aa_pKa_eff[aa] = pKa_eff(self.solution.ion_str, pKa, is_base, is_acid)
        return

    def get_net_charge(self):
        ch = self.solution.ch
        self.net_charge = 0
        self.net_neg_charge = 0
        self.net_pos_charge = 0
        for aa in self.aa_counts.keys():
            n       = self.aa_counts[aa]
            pKa_eff = self.aa_pKa_eff[aa]
            is_base = con().is_base(aa)
            is_acid = con().is_acid(aa)
            term = get_titration_term(ch, pKa_eff, is_base, is_acid)
            self.net_charge += n*term
            if term < 0:
                self.net_neg_charge += n*term
            else:
                self.net_pos_charge += n*term
        return

    def get_charge_dens(self):
        self.charge_dens = self.net_charge/self.area * con().F/con().Na   # [C m-2]
        self.charge_dens_neg = self.net_neg_charge/self.area * con().F/con().Na   # [C m-2]
        self.charge_dens_pos = self.net_pos_charge/self.area * con().F/con().Na   # [C m-2]
        return

    def get_charge_dens_dl(self):
        self.charge_dens_dl = self.solution.m_z*con().e*self.charge_dens*self.radius/\
                                (con().kT*self.solution.eps*con().eps0)
        return

    def get_theta_0(self):
        def residual(theta_p0):
            lhs = self.charge_dens_dl*(1.0-(np.tanh(theta_p0/4.0))**2)
            rhs = theta_p0*(1.0-(np.tanh(theta_p0/4.0))**2)\
                    + 4.0*self.solution.kappa*self.radius*np.tanh(theta_p0/4.0)
            return lhs - rhs

        fit = optimize.least_squares(residual, x0=[1.0], bounds=(-100, 100), ftol=1e-12)
        assert fit.success
        assert fit.fun[0] < 1e-6
        self.theta_0 = fit.x[0]
        return

    def get_fp(self):
        kap_r = self.solution.kappa*self.radius
        num   = 1 - 1/kap_r + (1 + 1/kap_r)*np.exp(-2.0*kap_r)
        denom = 1 + 1/kap_r - (1 + 1/kap_r)*np.exp(-2.0*kap_r)
        # 2016 Guelat:  1/kap_r is (+) in denom {agrees with Hsu and Liu 2009}
        # I believe 2012 Guelat is incorrect
        self.fp = num/denom
        return


# Functions called by objects (during construction)_____________________________

def pKa_eff(ion_str, pKa, is_base, is_acid):
    if is_base:
        valence = 1
    elif is_acid:
        valence = 0
    a_debye = 0.5114
    B       = 0.1
    return pKa + 2*(valence-1)*(a_debye*np.sqrt(ion_str)/(1+np.sqrt(ion_str))-B*ion_str)

def get_titration_term(ch, pKa_eff, is_base, is_acid):
    Ka_eff = 10**(-1.0*pKa_eff)
    if is_acid:
        return -1.0/(1.0+ch/Ka_eff)
    elif is_base:
        return 1.0/(1.0+Ka_eff/ch)

def capital_theta(obj):
    capital_theta = 4.0*np.exp(obj.solution.kappa*obj.dpr/2.0)\
                    *np.arctanh(np.exp(-1.0*obj.solution.kappa*obj.dpr/2.0)\
                    *np.tanh(obj.theta_0/4.0))
    return capital_theta


# Functions for solving the problem_____________________________________________

def delta_g_elec(protein, resin):
    # Electorstatic delta G [J mol-1]
    cap_theta_p = protein.capital_theta
    fp          = protein.fp
    cap_theta_r = resin.capital_theta
    solution    = protein.solution
    kappa       = solution.kappa
    dpr         = protein.dpr

    constant = con().Na*con().eps0*solution.eps\
                *(con().kT**2)*np.pi*protein.radius/((solution.m_z*con().e)**2)
    if fp < 0.0:
        variable_term = np.arctan(np.sqrt(abs(fp))*np.exp(-kappa*dpr))
    else:
        variable_term = np.arctanh(np.sqrt(fp)*np.exp(-kappa*dpr))

    if protein.negative_sign:
        sign = -1.0
    else:
        sign = 1.0
    bracket = sign*(cap_theta_p**2 + fp*cap_theta_r**2)/fp * np.log(1-fp*np.exp(-2.0*kappa*dpr))\
                + 4*cap_theta_p*cap_theta_r/np.sqrt(abs(fp))*variable_term
    # NB:  The theoretically correct formulation has a negative sign on the entire first term
    return constant*bracket

def delta_g_vdw(protein):
    # VdW delta G [J mol-1]
    radius  = protein.radius
    dpr     = protein.dpr
    result = -1.0*con().hamaker*con().Na/6.0*(radius/dpr + radius/(2*radius+dpr)\
                + np.log(dpr/(2*radius+dpr)))
    return result

def delta_g(protein, resin):
    # [J mol-1]
    g_elec = delta_g_elec(protein, resin)
    g_vdw  = delta_g_vdw(protein)
    return g_elec + g_vdw

def get_Keq_dim(protein, resin):
    g = delta_g(protein, resin)
    mass = protein.mass*con().kg_per_Da
    result = con().h/np.sqrt(2.0*np.pi*mass*con().kT)\
                *(np.exp(-1.0*g/(con().Na*con().kT))-1.0)
    return result

def get_kprime(surf_area, porosity, protein, resin):
    keq_dim = get_Keq_dim(protein, resin)
    kprime  = base_classes.get_kprime(surf_area, porosity, keq_dim)
    return kprime

def get_Keq_nd(protein, resin, phi_inv_m):
    keq_dim = get_Keq_dim(protein, resin)
    keq_nd = base_classes.get_Keq_nd(phi_inv_m, keq_dim)
    return keq_nd
