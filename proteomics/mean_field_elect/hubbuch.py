import base_classes
from base_classes import constants as con

import numpy as np
from scipy import optimize, integrate


# Classes_______________________________________________________________________

class resin(base_classes.resin):
    """Resin in Hubbuch model"""
    def __init__(self, solution, ligand, surf_dens, name='resin_name'):
        # Note to self:  constructors aren't generally needed for subclasses
        super().__init__(solution, ligand, surf_dens, name)
        self.get_psi0()
        self.get_cap_in()
        self.p = get_p(self)
        return

    def get_psi0(self):
        """Solve for surface potential"""
        def residual(psi0):
            """Charge density [C m-2] of the inner layer at surface potential psi"""
            total = self.surf_dens * get_titration_term(self.solution.ch, self.pKa,
                                                        self.is_base, self.is_acid, psi0)
            lhs = con().e * con().Na * total
            rhs = get_charge_dens_diffuse(psi0, self)
            return lhs - rhs

        fit = optimize.least_squares(residual, x0=[0.0], bounds=(-1, 1), ftol=1e-12)
        assert fit.success
        if fit.cost > 1e-6:
            print('resin')
            print(fit)
            assert fit.fun[0] < 1e-6
        self.psi0 = fit.x[0]
        return

    def get_cap_in(self):
        """Inner layer capacitance"""
        total = self.surf_dens * get_titration_term_capacitance(self.solution.ch,
                                    self.pKa, self.is_base, self.is_acid, self.psi0)
        self.cap_in = -1.0 * con().e * con().Na * total
        return


class protein(base_classes.protein):
    """Protein in Hubbuch model"""
    def __init__(self, solution, seq, name='protein_name'):
        super().__init__(solution, seq, name)
        self.get_surf_dens()
        self.get_psi0()
        self.get_cap_in()
        self.p = get_p(self)
        return

    def get_surf_dens(self):
        """Compute ligand density [mol m-2] for titratable protein groups"""
        self.surf_dens = {}
        for aa in self.aa_counts.keys():
            self.surf_dens[aa] = self.aa_counts[aa]/(self.area * con().Na) # [mol m-2]
        return

    def get_psi0(self):
        """Solve for surface potential"""
        def residual(psi0):
            """Charge density [C m-2] of the inner layer at surface potential psi"""
            total = 0
            for aa in self.aa_counts.keys():
                pKa, is_base, is_acid = con().get_chemical_info(aa)
                total += self.surf_dens[aa] * get_titration_term(self.solution.ch, pKa, is_base,
                                                                is_acid, psi0)
            lhs = con().e * con().Na * total
            rhs = get_charge_dens_diffuse(psi0, self)
            return lhs - rhs

        fit = optimize.least_squares(residual, x0=[0.0], bounds=(-1, 1), ftol=1e-12)
        assert fit.success
        if fit.cost > 1e-6:
            print('protein')
            print(fit)
            assert fit.fun[0] < 1e-6
        self.psi0 = fit.x[0]
        return

    def get_cap_in(self):
        """Inner layer capacitance"""
        total = 0
        for aa in self.aa_counts.keys():
            pKa, is_base, is_acid = con().get_chemical_info(aa)
            total += self.surf_dens[aa] * get_titration_term_capacitance(self.solution.ch,
                                                        pKa, is_base, is_acid, self.psi0)
        self.cap_in = -1.0 * con().e * con().Na * total
        return


# Functions called by objects (during construction)__________________________________

def get_charge_dens_diffuse(psi0, obj):
    """Diffuse layer charge density [C m-2] at surface potential psi0"""
    return obj.solution.eps*con().eps0*obj.solution.kappa*psi0 # linear
    # return 2*solution.eps*eps0*solution.kappa*kT/e*np.sinh(e*psi0/(2*kT)) # nonlinear

def get_titration_term(ch, pKa, is_base, is_acid, psi0):
    """For charge density"""
    Ka = 10**(-1.0*pKa)
    if is_acid:
        return -1.0/(1.0 + (ch/Ka) * np.exp(-1.0*con().e*psi0/con().kT) )
    elif is_base:
        return 1.0/(1.0 + (Ka/ch) * np.exp(con().e*psi0/con().kT) )

def get_titration_term_capacitance(ch, pKa, is_base, is_acid, psi0):
    """For capicitance, from the gradient of charge density with respect to potential"""
    Ka = 10**(-1.0*pKa)
    if is_acid:
        num = -1.0 * ch/Ka * con().e/con().kT * np.exp(-1.0*con().e*psi0/con().kT)
        den = ( 1 + ch/Ka * np.exp(-1.0*con().e*psi0/con().kT) )**2
    elif is_base:
        num = -1.0 * Ka/ch * con().e/con().kT * np.exp(con().e*psi0/con().kT)
        den = ( 1.0 + Ka/ch * np.exp(con().e*psi0/con().kT) )**2
    return num/den

def get_p(obj):
    """Regularization parameter"""
    return obj.solution.cap_dif/(obj.solution.cap_dif + obj.cap_in)


# Functions for solving the problem_____________________________________________

def get_ener(z, protein, resin, scale=1e10):
    """Get energy for the surface separation z [m, when scale=1].
        Reference Carnie and Chan's (1993) original formulation.
        Scale is used to increase integration accuracy in functions below.
        E.g. set scale=1e10 when inputting z in Angstroms"""
    z *= 1.0/scale      # Scale down from Angstroms
    kappa = protein.solution.kappa
    a1 = protein.radius
    psi1 = protein.psi0
    psi2 = resin.psi0
    p1 = protein.p
    p2 = resin.p
    g = (1-2*p1)*(1-2*p2)

    if g < 0:
        h = np.arctan(np.sqrt(-1.0*g)*np.exp(-1*kappa*z))
    else:
        h = np.arctanh(np.sqrt(g)*np.exp(-1*kappa*z))

    if h == np.nan or h == np.inf:
        raise ValueError('NaN issue. g = {.:2f}, h = {.:2f}'.format(g, h))
    elif 1-g*np.exp(-2*kappa*z) < 0:
        raise ValueError('g = {:.2f}, kappa = {:.2e}, z = {:.2e} \n p_prot = {:.2f}\
                        p_res = {:.2f}'.format(g, kappa, z, p1, p2))
    elif g == 0.0:
        raise ValueError('g = 0 issue')
    elif z < 0:
        raise ValueError('z < 0 issue. z = {:.2e}'.format(z))
    else:
        ener = 2*np.pi*protein.solution.eps*con().eps0*a1*(((1-2*p1)*psi2**2 + (1-2*p2)*psi1**2)\
                    /(2*g)*np.log(1-g*np.exp(-2*kappa*z)) + 2*psi1*psi2/np.sqrt(np.abs(g))*h)
    return ener

def find_ener_min_x(protein, resin, scale=1e10, x_up_bnd_coeff=1e-8):
    """Find the separation distance at the energy minimum"""
    x_up_bnd = scale * x_up_bnd_coeff
    fit = optimize.minimize_scalar(get_ener, args=(protein, resin, scale),
                                    method='bounded', bracket=(1e-13*scale, x_up_bnd),
                                    bounds=(1e-13, x_up_bnd),
                                    options={'maxiter':10000})
    if not fit.success:
        print(fit)
        assert fit.success
    return fit.x

def get_integrand(z, protein, resin, scale=1):
    """Equation 15, for dimensional Keq"""
    return np.exp(-1.0*get_ener(z, protein, resin, scale)/con().kT) - 1

def get_Keq_dim_integrate(protein, resin, scale=1e10, x_up_bnd_coeff=1e-8,
                            dpr_coeff=2e-10, find_min=False):
    """Keq in [m]"""
    x_up_bnd = scale * x_up_bnd_coeff
    if find_min:
        x_lo_bnd = find_ener_min_x(protein, resin, scale, x_up_bnd_coeff)
    else:
        x_lo_bnd = scale * dpr_coeff
    res = integrate.quad(get_integrand, x_lo_bnd,  x_up_bnd, args=(protein, resin, scale))
    if abs(res[1]/res[0]) > 1e-6:
        # print(res)
        assert abs(res[1]/res[0]) < 1e-6
    return 1.0/scale * res[0]

def get_kprime(surf_area, porosity, protein, resin, scale=1e10,
                x_up_bnd_coeff=1e-8, dpr_coeff=2e-10):
    keq_dim = get_Keq_dim_integrate(protein, resin, scale, x_up_bnd_coeff, dpr_coeff)
    kprime  = base_classes.get_kprime(surf_area, porosity, keq_dim)
    return kprime

def get_Keq_nd(protein, resin, phi_inv_m, dpr_coeff=2e-10):
    keq_dim = get_Keq_dim_integrate(protein, resin, dpr_coeff=dpr_coeff)
    keq_nd = base_classes.get_Keq_nd(phi_inv_m, keq_dim)
    return keq_nd
