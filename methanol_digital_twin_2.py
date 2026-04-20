#!/usr/bin/env python3
# =============================================================================
#  DIGITAL TWIN: ISOTHERMAL MULTI-TUBULAR METHANOL SYNTHESIS REACTOR
#  PhD-Level Implementation — Extended Reaction Network
# =============================================================================
#
#  REACTION NETWORK (7 reactions, 10 species):
#   R1: CO  + 2H2   <=> CH3OH              dH = -90.7  kJ/mol
#   R2: CO2 + 3H2   <=> CH3OH + H2O       dH = -49.5  kJ/mol
#   R3: CO  + H2O   <=> CO2  + H2         dH = -41.2  kJ/mol  [WGS]
#   R4: 2CH3OH      <=> DME  + H2O        dH = -23.4  kJ/mol  [DME]
#   R5: CO  + 3H2   --> CH4  + H2O        dH = -206.2 kJ/mol  [Methanation]
#   R6: 2CO + 4H2   <=> EtOH + H2O        dH = -253.6 kJ/mol  [Ethanol]
#   R7: 3CO + 6H2   <=> PrOH + 2H2O       dH = -417.3 kJ/mol  [1-Propanol]
#
#  SPECIES INDEX:
#   0=CO, 1=CO2, 2=H2, 3=H2O, 4=MeOH, 5=DME, 6=CH4, 7=EtOH, 8=PrOH, 9=N2
#
#  KEY REFERENCES:
#   [GRA88]  Graaf et al., Chem. Eng. Sci. 43 (1988) 3185-3195  [Groningen, NL]
#   [GRA90]  Graaf et al., Chem. Eng. Sci. 45 (1990) 773-783    [Groningen, NL]
#   [VBF96]  Vanden Bussche & Froment, J. Catal. 161 (1996) 1-10 [Ghent, BE]
#   [NEP20]  Nestler et al., Chem. Eng. J. 394 (2020) 124881    [KIT, DE]
#   [SEI18]  Seidel et al., Chem. Eng. Sci. 175 (2018) 130-138  [OvGU, DE]
#   [BOZ16]  Bozzano & Manenti, PECS 56 (2016) 71-105           [PoliMi, IT]
#   [BL92]   Bercic & Levec, IECR 31 (1992) 1035-1040           [Ljubljana/NL]
#   [KOP10]  Kopyscinski et al., Fuel 89 (2010) 1763-1783        [ETH/PSI, CH]
#   [TRO92]  Tronconi et al., CES 47 (1992) 2227-2232            [PoliMi, IT]
#   [PAR14]  Park et al., Fuel 118 (2014) 202-213               [MegaMax]
#   [ZS70]   Zehner & Schlünder, CIT 42 (1970) 933              [KIT, DE]
#   [MAN13]  Manenti et al., CEJ 228 (2013) 1-8                 [PoliMi, IT]
#
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: PHYSICAL CONSTANTS & MOLECULAR PROPERTIES
# =============================================================================

R_GAS = 8.314          # J/(mol·K)  universal gas constant
R_BAR = 8.314e-5       # bar·m³/(mol·K)

# Species names and molecular weights [g/mol]
SPECIES = ['CO', 'CO2', 'H2', 'H2O', 'MeOH', 'DME', 'CH4', 'EtOH', 'PrOH', 'N2']
MW = np.array([28.010, 44.010, 2.016, 18.015, 32.042, 46.068, 16.043, 46.068, 60.095, 28.014])
N_SPECIES = len(SPECIES)

# Stoichiometric matrix [species x reactions]
# Rows: CO CO2 H2 H2O MeOH DME CH4 EtOH PrOH N2
# Cols: R1  R2  R3  R4  R5  R6  R7
STOICH = np.array([
    [-1,  0, -1,  0, -1, -2, -3],   # CO
    [ 0, -1, +1,  0,  0,  0,  0],   # CO2
    [-2, -3, +1,  0, -3, -4, -6],   # H2
    [ 0, +1, -1, +1, +1, +1, +2],   # H2O
    [+1, +1,  0, -2,  0,  0,  0],   # MeOH
    [ 0,  0,  0, +1,  0,  0,  0],   # DME
    [ 0,  0,  0,  0, +1,  0,  0],   # CH4
    [ 0,  0,  0,  0,  0, +1,  0],   # EtOH
    [ 0,  0,  0,  0,  0,  0, +1],   # PrOH
    [ 0,  0,  0,  0,  0,  0,  0],   # N2
], dtype=float)

# Standard heats of reaction at 298.15 K [J/mol]
DHR_298 = np.array([-90700., -49500., -41200., -23400., -206200., -253600., -417300.])

# Lennard-Jones parameters for viscosity/diffusivity [sigma in Angstrom, eps/k in K]
# Source: Poling, Prausnitz & O'Connell, 5th Ed.
LJ_SIGMA = {'CO': 3.690, 'CO2': 3.941, 'H2': 2.827, 'H2O': 2.641,
            'MeOH': 3.626, 'DME': 4.307, 'CH4': 3.758, 'EtOH': 4.530,
            'PrOH': 4.549, 'N2': 3.798}
LJ_EPSK  = {'CO': 91.7,  'CO2': 195.2, 'H2': 59.7,  'H2O': 809.1,
            'MeOH': 481.8,'DME': 395.0,'CH4': 148.6,'EtOH': 391.0,
            'PrOH': 412.0,'N2': 71.4}

# Critical properties for PR-EOS [Tc/K, Pc/bar, omega]
CRIT = {
    'CO':   (132.85, 34.94, 0.0480),
    'CO2':  (304.12, 73.74, 0.2239),
    'H2':   ( 33.19, 13.13, -0.2160),
    'H2O':  (647.10,220.64, 0.3449),
    'MeOH': (512.64, 80.97, 0.5625),
    'DME':  (400.10, 53.70, 0.2003),
    'CH4':  (190.56, 45.99, 0.0115),
    'EtOH': (514.71, 63.84, 0.6455),
    'PrOH': (536.80, 51.75, 0.6233),
    'N2':   (126.19, 33.96, 0.0372),
}

# Shomate coefficients for Cp [J/(mol·K)] — Source: NIST WebBook
# Form: Cp = A + B*t + C*t^2 + D*t^3 + E/t^2  where t = T[K]/1000
SHOMATE = {
    # sp:      A         B          C          D          E
    'CO':   ( 25.5676,   6.0961,    4.0546,   -2.6713,    0.1310),
    'CO2':  ( 24.9974,  55.1870,  -33.6914,    7.9484,   -0.1366),
    'H2':   ( 33.0662, -11.3634,   11.4328,   -2.7729,   -0.1586),
    'H2O':  ( 30.0920,   6.8325,    6.7934,   -2.5345,    0.0821),
    'MeOH': ( 14.1089,  97.9293,   -9.6696,   -0.0790,    0.2395),
    'DME':  ( 17.0380, 178.380,   -68.100,     8.860,    -0.150 ),
    'CH4':  ( -0.7030, 108.477,   -42.521,     5.862,     0.679 ),
    'EtOH': ( -0.2947, 178.630,  -100.380,    22.136,     0.210 ),
    'PrOH': ( 12.518,  274.140,  -162.940,    36.420,    -0.112 ),
    'N2':   ( 26.0929,   8.2148,   -1.9764,    0.1592,    0.0444),
}


# =============================================================================
# SECTION 2: THERMODYNAMIC MODEL
# =============================================================================

class ThermoModel:
    """
    Thermodynamic calculations: Cp, dHr(T), equilibrium constants.

    References:
      - NIST WebBook for Shomate coefficients
      - Graaf et al. (1988) for Keq correlations [Groningen, NL]
      - Stull, Westrum & Sinke for DME Keq
    """

    @staticmethod
    def cp_species(sp: str, T: float) -> float:
        """
        Molar heat capacity via Shomate equation [J/(mol·K)].
        T in Kelvin.
        """
        A, B, C, D, E = SHOMATE[sp]
        t = T / 1000.0
        return A + B*t + C*t**2 + D*t**3 + E/t**2

    @staticmethod
    def cp_mix(y: np.ndarray, T: float) -> float:
        """
        Molar heat capacity of gas mixture [J/(mol·K)].
        y = mole fractions array (length N_SPECIES)
        """
        cp = sum(y[i] * ThermoModel.cp_species(SPECIES[i], T) for i in range(N_SPECIES))
        return cp

    @staticmethod
    def enthalpy_sensible(sp: str, T: float, T_ref: float = 298.15) -> float:
        """
        Sensible enthalpy H(T) - H(T_ref) [J/mol] using Shomate integral.
        """
        A, B, C, D, E = SHOMATE[sp]
        def H_shomate(Tk):
            t = Tk / 1000.0
            return (A*t + B*t**2/2 + C*t**3/3 + D*t**4/4 - E/t) * 1000.0  # J/mol
        return H_shomate(T) - H_shomate(T_ref)

    @staticmethod
    def dHr_T(rxn_idx: int, T: float) -> float:
        """
        Temperature-dependent reaction enthalpy [J/mol].
        Uses Kirchhoff's law: dHr(T) = dHr(298) + integral(dCp)dT
        """
        nu = STOICH[:, rxn_idx]
        dH = DHR_298[rxn_idx]
        for i, sp in enumerate(SPECIES):
            if nu[i] != 0:
                dH += nu[i] * ThermoModel.enthalpy_sensible(sp, T)
        return dH

    @staticmethod
    def keq(rxn_idx: int, T: float) -> float:
        """
        Equilibrium constants.

        R1 (CO+2H2<=>MeOH):       from Graaf 1988 [bar^-2]
        R2 (CO2+3H2<=>MeOH+H2O):  from Graaf 1988 [bar^-2]
        R3 (WGS CO+H2O<=>CO2+H2): from Graaf 1988 [dimensionless]
        R4 (2MeOH<=>DME+H2O):     from Stull, Westrum & Sinke [-]
        R5 (Methanation):          very large (irreversible)
        R6 (Ethanol):              van't Hoff [bar^-3]
        R7 (1-Propanol):           ASF-derived
        """
        if rxn_idx == 0:   # R1: CO+2H2<=>MeOH
            # Graaf (1988) Eq. A1
            return 10.0 ** (5139.0/T - 12.621)    # bar^-2

        elif rxn_idx == 1: # R2: CO2+3H2<=>MeOH+H2O
            # Graaf (1988) Eq. A2
            return 10.0 ** (3066.0/T - 10.592)    # bar^-2

        elif rxn_idx == 2: # R3: WGS
            # Graaf (1988) Eq. A3  — note sign: CO+H2O->CO2+H2
            return 10.0 ** (2073.0/T - 2.029)     # dimensionless

        elif rxn_idx == 3: # R4: 2MeOH<=>DME+H2O
            # From Stull, Westrum & Sinke thermochemical tables
            # dG = -RT ln(K), dH = -23400 J/mol, dS ≈ +15 J/(mol·K)
            return np.exp(4019.5/T - 3.707)       # dimensionless

        elif rxn_idx == 4: # R5: Methanation (irreversible)
            return 1e30                            # effectively irreversible

        elif rxn_idx == 5: # R6: Ethanol 2CO+4H2<=>EtOH+H2O
            # Van't Hoff approx, dH = -253600 J/mol
            return np.exp(30530.0/T - 49.8)       # bar^-3

        elif rxn_idx == 6: # R7: 1-Propanol 3CO+6H2<=>PrOH+2H2O
            # From ASF + thermodynamic consistency
            return np.exp(50200.0/T - 85.0)       # bar^-5

        return 1.0


# =============================================================================
# SECTION 3: TRANSPORT PROPERTIES MODEL
# =============================================================================

class TransportModel:
    """
    Gas-phase transport properties: viscosity, thermal conductivity, diffusivity.

    References:
      - Chapman-Enskog theory: Poling, Prausnitz & O'Connell (2001)
      - Wilke mixing rule: Wilke (1950), J. Chem. Phys. 18, 517
      - Neufeld collision integrals: Neufeld et al. (1972), J. Chem. Phys. 57, 1100
      - Zehner & Schlunder (1970) [KIT, Germany] for effective bed conductivity
    """

    @staticmethod
    def omega_mu(Tstar: float) -> float:
        """
        Neufeld collision integral for viscosity/diffusivity.
        Neufeld et al. (1972), J. Chem. Phys. 57, 1100.
        """
        A, B, C, D, E, F, G, H = (1.16145, 0.14874, 0.52487,
                                    0.77320, 2.16178, 2.43787,
                                    6.43500e-4, 7.27371)
        return A/Tstar**B + C/np.exp(D*Tstar) + E/np.exp(F*Tstar) + G*np.sin(H*Tstar - 1.069)

    @staticmethod
    def viscosity_pure(sp: str, T: float) -> float:
        """
        Pure component viscosity via Chapman-Enskog [Pa·s].
        mu = 2.6693e-6 * sqrt(M*T) / (sigma^2 * Omega_mu)
        """
        sigma = LJ_SIGMA[sp]
        epsk  = LJ_EPSK[sp]
        M     = MW[SPECIES.index(sp)]
        Tstar = T / epsk
        omega = TransportModel.omega_mu(Tstar)
        return 2.6693e-6 * np.sqrt(M * T) / (sigma**2 * omega)

    @staticmethod
    def viscosity_mix(y: np.ndarray, T: float) -> float:
        """
        Mixture viscosity via Wilke mixing rule [Pa·s].
        Wilke (1950), J. Chem. Phys. 18, 517.
        """
        mu = np.array([TransportModel.viscosity_pure(sp, T) for sp in SPECIES])
        phi = np.zeros((N_SPECIES, N_SPECIES))
        for i in range(N_SPECIES):
            for j in range(N_SPECIES):
                ratio_mu = mu[i]/mu[j] if mu[j] > 0 else 1.0
                ratio_M  = MW[j]/MW[i]
                phi[i,j] = (1 + ratio_mu**0.5 * ratio_M**0.25)**2 / np.sqrt(8*(1 + MW[i]/MW[j]))
        mu_mix = sum(y[i]*mu[i] / sum(y[j]*phi[i,j] for j in range(N_SPECIES))
                     for i in range(N_SPECIES) if y[i] > 1e-12)
        return max(mu_mix, 1e-6)

    @staticmethod
    def conductivity_pure(sp: str, T: float) -> float:
        """
        Pure component thermal conductivity via modified Eucken correlation [W/(m·K)].
        Reid, Prausnitz & Poling (1987).
        """
        mu = TransportModel.viscosity_pure(sp, T)
        cp = ThermoModel.cp_species(sp, T)
        M  = MW[SPECIES.index(sp)]
        # Modified Eucken: lambda = mu * Cp * (1.32 + 1.77*R/Cp)
        # Simplified: lambda = mu * Cp / M * (Cp*M/R + 1.25) / (Cp*M/R + 0.565)
        gamma = cp / (cp - R_GAS)  # approximate
        f_int = max(0.8, 1.0 + 0.6 * (gamma - 1.0))  # interpolation
        return mu * cp / M * f_int

    @staticmethod
    def conductivity_mix(y: np.ndarray, T: float) -> float:
        """
        Mixture thermal conductivity via Lindsay-Bromley approximation [W/(m·K)].
        Simple Wilke-type mixing for conductivity.
        """
        lam = np.array([TransportModel.conductivity_pure(sp, T) for sp in SPECIES])
        mu  = np.array([TransportModel.viscosity_pure(sp, T) for sp in SPECIES])
        # Use Wilke-type formula with A_ij based on viscosities
        lam_mix = 0.0
        denom = np.zeros(N_SPECIES)
        for i in range(N_SPECIES):
            s = 0.0
            for j in range(N_SPECIES):
                phi_ij = (1 + (mu[i]/max(mu[j],1e-10))**0.5 * (MW[j]/MW[i])**0.25)**2 / \
                          np.sqrt(8*(1 + MW[i]/MW[j]))
                s += y[j] * phi_ij
            if s > 0 and y[i] > 1e-12:
                lam_mix += y[i] * lam[i] / s
        return max(lam_mix, 0.01)

    @staticmethod
    def diffusivity_binary(sp_i: str, sp_j: str, T: float, P_bar: float) -> float:
        """
        Binary diffusivity via Chapman-Enskog [m^2/s].
        P in bar. Poling et al. (2001) Eq. 11-3.1.
        """
        Mi, Mj = MW[SPECIES.index(sp_i)], MW[SPECIES.index(sp_j)]
        sig_ij = 0.5 * (LJ_SIGMA[sp_i] + LJ_SIGMA[sp_j])
        eps_ij = np.sqrt(LJ_EPSK[sp_i] * LJ_EPSK[sp_j])
        Tstar  = T / eps_ij
        omega  = TransportModel.omega_mu(Tstar)
        Mij    = 2.0 / (1.0/Mi + 1.0/Mj)  # harmonic mean
        Dij    = 1.883e-3 * T**1.5 * np.sqrt(1.0/Mi + 1.0/Mj) / (P_bar * sig_ij**2 * omega)
        return max(Dij * 1e-4, 1e-8)  # m^2/s

    @staticmethod
    def diffusivity_mix(sp_i: str, y: np.ndarray, T: float, P_bar: float) -> float:
        """
        Effective mixture diffusivity for species i in mixture [m^2/s].
        Wilke-Lee modification of Stefan-Maxwell.
        """
        idx_i = SPECIES.index(sp_i)
        yi    = y[idx_i]
        sum_j = 0.0
        for j, sp_j in enumerate(SPECIES):
            if j != idx_i and y[j] > 1e-12:
                Dij    = TransportModel.diffusivity_binary(sp_i, sp_j, T, P_bar)
                sum_j += y[j] / Dij
        if sum_j < 1e-30:
            return 1e-5
        return (1.0 - yi) / sum_j


# =============================================================================
# SECTION 4: KINETICS MODELS — LHHW LIBRARY
#
#   Selectable models (pass model='...' to KineticsModel or DigitalTwin):
#     'graaf'  — Graaf et al. (1988/1990) + Nestler (2020) / Seidel (2018) [default]
#     'vbf'    — Vanden Bussche & Froment (1996), CO₂-only primary pathway
#     'park'   — Park et al. (2014), Graaf form re-fitted for commercial MegaMax
#
#   All rates in [mol/(kg_cat·s)], fugacities in [bar].
#   Rate constant:   k(T) = A  * exp(-Ea  / (R·T))
#   Adsorption coef: K(T) = B  * exp(+ΔH / (R·T))   [ΔH > 0: exothermic ads. convention]
# =============================================================================


class KineticsBase:
    """
    Abstract base class — uniform interface for all kinetics models.
    Sub-classes must implement: rates(T, f) and rate_constants(T).
    """
    MODEL_NAME = "Base"
    MODEL_REF  = ""

    def rates(self, T: float, f: np.ndarray) -> np.ndarray:
        """Return 7 reaction rates [mol/(kg_cat·s)] given T [K] and fugacities f [bar]."""
        raise NotImplementedError

    def rate_constants(self, T: float) -> dict:
        """Return dict of temperature-evaluated rate and adsorption constants."""
        raise NotImplementedError


class KineticsGraaf(KineticsBase):
    """
    Graaf et al. (1988/1990) LHHW kinetics — 3 CO/CO₂ primary reactions.
    Parameters calibrated for MegaMax Cu/ZnO/Al₂O₃ using:
      Nestler et al. (2020) [KIT, DE] — extended T/P range
      Seidel et al. (2018) [OvGU, DE] — MegaMax-specific refinement
      Park et al. (2014)   [—, KR]    — industrial catalyst validation

    All rates in [mol/(kg_cat·s)], fugacities in [bar].
    Reactions handled: R1–R3 (Graaf) + R4 (Bercic/Levec) + R5 (Kopyscinski)
                       + R6 (Tronconi) + R7 (ASF/Bozzano)

    References:
      [GRA88]  Graaf et al., CES 43 (1988) 3185-3195
      [GRA90]  Graaf et al., CES 45 (1990) 773-783
      [NEP20]  Nestler et al., CEJ 394 (2020) 124881
      [SEI18]  Seidel et al., CES 175 (2018) 130-138
      [BL92]   Bercic & Levec, IECR 31 (1992) 1035-1040
      [KOP10]  Kopyscinski et al., Fuel 89 (2010) 1763-1783
      [TRO92]  Tronconi et al., CES 47 (1992) 2227-2232
      [BOZ16]  Bozzano & Manenti, PECS 56 (2016) 71-105
    """
    MODEL_NAME = "Graaf (1988/1990) + Nestler (2020)"
    MODEL_REF  = ("Graaf et al. CES 43 (1988) 3185 | "
                  "Nestler et al. CEJ 394 (2020) 124881")

    def __init__(self, params: dict = None):
        """
        Initialize with kinetic parameters.
        Default: Graaf (1988) + Nestler (2020) / Seidel (2018) MegaMax calibration.
        Pass custom params dict to override for your catalyst/conditions.
        """
        self.p = params if params is not None else self._default_params()

    @staticmethod
    def _default_params() -> dict:
        """
        Kinetic parameters from Graaf (1988), Bercic-Levec (1992),
        Nestler (2020), Bozzano-Manenti (2016).

        Units:
          k_i: mol / (kg_cat * s * bar^n)  where n depends on reaction
          K_j: bar^-1  (adsorption constants)
        """
        return {
            # ---- R1: CO hydrogenation [Graaf 1988, Table 4] ----
            # Graaf (1988) fitted at 210-245°C, 15-50 bar.
            # At 260°C / 75 bar, the driving force is amplified — empirical calibration
            # factor of ~50x applied (recommended by Nestler 2020 for extrapolation).
            # NOTE: calibrate with .calibrate() using your plant data.
            'A1':    0.0818,     # mol/(kg_cat·s·bar^1.5)  [Graaf 1988 * 1000 / 50]
            'Ea1':   40400.,     # J/mol

            # ---- R2: CO2 hydrogenation [Graaf 1988] ----
            'A2':    21.82,      # mol/(kg_cat·s·bar^1.5)  [Graaf 1988 * 1000 / 50]
            'Ea2':   17130.,     # J/mol

            # ---- R3: WGS [Graaf 1988] — thermally activated, high Ea ----
            'A3':    1.536e9,    # mol/(kg_cat·s·bar^1.5)  [Graaf 1988 * 1000 / 50]
            'Ea3':   124100.,    # J/mol

            # ---- Adsorption constants for Graaf model ----
            'B_CO':   7.99e-7,   # bar^-1
            'dH_CO':  58100.,    # J/mol (positive => exothermic adsorption convention)
            'B_CO2':  1.02e-7,   # bar^-1
            'dH_CO2': 67400.,    # J/mol
            'B_Phi':  4.13e-11,  # bar^-0.5  (combined KH2O/KH2^0.5)
            'dH_Phi': 110900.,   # J/mol

            # ---- R4: DME [Bercic & Levec 1992, Bozzano & Manenti 2016] ----
            # DME forms primarily on acid sites — weaker on pure MegaMax than on bifunctional
            'A4':    3.7e2,      # mol/(kg_cat·s·bar^2)  [reduced for Cu/ZnO vs γ-Al2O3]
            'Ea4':   80000.,     # J/mol
            'B_MeOH': 5.39e-5,  # bar^-1
            'dH_MeOH': 70400.,  # J/mol

            # ---- R5: Methanation [Kopyscinski 2010, PSI/ETH Zurich] ----
            # Trace methanation on Cu/ZnO — significantly slower than MeOH synthesis
            'A5':    6.1e-1,     # mol/(kg_cat·s·bar^4)  [minor pathway on Cu/ZnO]
            'Ea5':   115000.,    # J/mol
            'B_CO_m': 1.2e-6,   # bar^-1
            'dH_CO_m': 70000.,  # J/mol

            # ---- R6: Ethanol [Tronconi 1992, PoliMi] ----
            # CO insertion mechanism — trace levels on Cu/ZnO industrial catalyst
            'A6':    2.5e-7,     # mol/(kg_cat·s·bar^4)
            'Ea6':   95000.,     # J/mol

            # ---- R7: 1-Propanol [ASF, Bozzano & Manenti 2016] ----
            'alpha0': 0.10,      # ASF chain growth factor pre-exp
            'E_alpha': 5000.,    # J/mol (weak T-dependence)
        }

    def rate_constants(self, T: float) -> dict:
        """Compute all temperature-dependent rate and adsorption constants."""
        p = self.p
        k = {}
        k['k1']    = p['A1']    * np.exp(-p['Ea1']    / (R_GAS * T))
        k['k2']    = p['A2']    * np.exp(-p['Ea2']    / (R_GAS * T))
        k['k3']    = p['A3']    * np.exp(-p['Ea3']    / (R_GAS * T))
        k['k4']    = p['A4']    * np.exp(-p['Ea4']    / (R_GAS * T))
        k['k5']    = p['A5']    * np.exp(-p['Ea5']    / (R_GAS * T))
        k['k6']    = p['A6']    * np.exp(-p['Ea6']    / (R_GAS * T))
        k['K_CO']  = p['B_CO']  * np.exp(+p['dH_CO']  / (R_GAS * T))
        k['K_CO2'] = p['B_CO2'] * np.exp(+p['dH_CO2'] / (R_GAS * T))
        k['Phi']   = p['B_Phi'] * np.exp(+p['dH_Phi'] / (R_GAS * T))  # KH2O/KH2^0.5
        k['K_MeOH']= p['B_MeOH']* np.exp(+p['dH_MeOH']/ (R_GAS * T))
        k['K_CO_m']= p['B_CO_m']* np.exp(+p['dH_CO_m']/ (R_GAS * T))
        k['alpha'] = p['alpha0']* np.exp(-p['E_alpha'] / (R_GAS * T))
        return k

    def rates(self, T: float, f: np.ndarray) -> np.ndarray:
        """
        Compute all 7 reaction rates [mol/(kg_cat * s)].

        Parameters
        ----------
        T   : temperature [K]
        f   : fugacities array [bar], index = species index

        Returns
        -------
        r   : array of 7 rates [mol/(kg_cat * s)]
              Positive = forward reaction
        """
        k = self.rate_constants(T)
        r = np.zeros(7)

        # Fugacities (pressures) for relevant species
        f_CO   = max(f[0], 0.0)
        f_CO2  = max(f[1], 0.0)
        f_H2   = max(f[2], 1e-8)
        f_H2O  = max(f[3], 0.0)
        f_MeOH = max(f[4], 0.0)
        f_DME  = max(f[5], 0.0)
        f_EtOH = max(f[7], 0.0)

        # Equilibrium constants
        Keq = [ThermoModel.keq(i, T) for i in range(7)]

        # --- Graaf denominator (shared by R1, R2, R3) ---
        # denominator = (1 + K_CO*fCO + K_CO2*fCO2) * (fH2^0.5 + Phi*fH2O)
        den_ads = (1.0 + k['K_CO']*f_CO + k['K_CO2']*f_CO2)
        den_h   = (f_H2**0.5 + k['Phi']*f_H2O)
        den     = den_ads * den_h
        den     = max(den, 1e-30)

        # ---- R1: CO + 2H2 <=> MeOH  [Graaf 1988] ----
        # r1 = k1*K_CO*(fCO*fH2^1.5 - fMeOH/(fH2^0.5*Keq1)) / den
        DrivingForce1 = f_CO*f_H2**1.5 - f_MeOH / (max(f_H2**0.5, 1e-8) * Keq[0])
        r[0] = k['k1'] * k['K_CO'] * DrivingForce1 / den

        # ---- R2: CO2 + 3H2 <=> MeOH + H2O  [Graaf 1988] ----
        # r2 = k2*K_CO2*(fCO2*fH2^1.5 - fMeOH*fH2O/(fH2^1.5*Keq2)) / den
        DrivingForce2 = f_CO2*f_H2**1.5 - f_MeOH*f_H2O / (max(f_H2**1.5, 1e-12) * Keq[1])
        r[1] = k['k2'] * k['K_CO2'] * DrivingForce2 / den

        # ---- R3: WGS  CO + H2O <=> CO2 + H2  [Graaf 1988] ----
        # r3 = k3*K_CO2*(fCO2*fH2 - fCO*fH2O/Keq3) / den
        # NOTE: sign convention — positive r3 means CO2 + H2 produced (forward as written)
        DrivingForce3 = f_CO2*f_H2 - f_CO*f_H2O / Keq[2]
        r[2] = k['k3'] * k['K_CO2'] * DrivingForce3 / den

        # ---- R4: 2MeOH <=> DME + H2O  [Bercic & Levec 1992] ----
        # r4 = k4 * K_MeOH^2 * fMeOH^2 * (1 - fDME*fH2O/(Keq4*fMeOH^2)) /
        #      (1 + 2*(K_MeOH*fMeOH)^0.5)^4
        f_DME_eq = max(f_DME, 0.0)
        revDME = f_DME_eq * f_H2O / (Keq[3] * max(f_MeOH**2, 1e-20))
        DrivingForce4 = k['K_MeOH']**2 * f_MeOH**2 * (1.0 - revDME)
        den4 = max((1.0 + 2.0*(k['K_MeOH']*f_MeOH)**0.5)**4, 1e-30)
        r[3] = k['k4'] * DrivingForce4 / den4

        # ---- R5: Methanation  CO + 3H2 --> CH4 + H2O  [Kopyscinski 2010] ----
        # Treated as irreversible; small rate on Cu/ZnO at normal conditions
        den5 = max((1.0 + k['K_CO_m']*f_CO)**2, 1e-30)
        r[4] = k['k5'] * f_CO * f_H2**3 / den5

        # ---- R6: Ethanol  2CO + 4H2 <=> EtOH + H2O  [Tronconi 1992] ----
        revEtOH = f_EtOH * f_H2O / (Keq[5] * max(f_CO**2 * f_H2**4, 1e-40))
        r[5] = k['k6'] * k['K_CO']**2 * f_CO**2 * f_H2**2 * (1.0 - revEtOH) / max(den_ads**3, 1e-30)

        # ---- R7: 1-Propanol via chain growth  [Bozzano & Manenti 2016, ASF] ----
        alpha = min(k['alpha'], 0.5)   # chain growth probability (bounded)
        r[6] = alpha * abs(r[5])       # follows from C2 alcohol via CO insertion

        return r


# =============================================================================
# SECTION 4b: VANDEN BUSSCHE & FROMENT (1996) KINETICS
# =============================================================================

class KineticsVBF(KineticsBase):
    """
    Vanden Bussche & Froment (1996) kinetics — CO₂-only primary pathway.

    Physical interpretation: CO converts to methanol exclusively via
    reverse-WGS → CO₂ → MeOH route; direct CO hydrogenation (R1) is zero.

    PRIMARY REACTIONS handled internally:
      VBF-R1: CO₂ + 3H₂ ↔ CH₃OH + H₂O  [maps to stoich R2, idx=1]
      VBF-R2: CO₂ + H₂  ↔ CO   + H₂O   [reverse WGS → mapped to –R3, idx=2]

    Rate expressions (pressures/fugacities in bar):
      r_MeOH = k5* · f_CO₂ · f_H₂ · (1 – f_H₂O·f_MeOH / (K_eq1 · f_H₂³·f_CO₂)) / den³
      r_RWGS = k1* · f_CO₂ · (1 – f_CO·f_H₂O / (K_RWGS · f_H₂·f_CO₂))         / den
      den    = 1 + K₃·(f_H₂O/f_H₂) + √K_H₂·f_H₂^0.5 + K_H₂O·f_H₂O

    Adsorption sign convention (VBF 1996):
      √K_H₂  = B · exp(–dH / RT)   [H₂ dissociative step, endothermic from surface]
      K_H₂O  = B · exp(+dH / RT)   [H₂O adsorption, strongly exothermic]
      K₃     ≡ K_H₂O/(K₈·K₉·√K_H₂) treated as constant (weak T-dep. over 210–260°C)

    ⚠  VALIDATED RANGE: 210–260°C, 15–51 bar (Vanden Bussche & Froment 1996).
       At high feed-H₂O (f_H₂O > 1 bar) the strong K₃·(H₂O/H₂) inhibition term
       significantly suppresses rates — consistent with the VBF physical picture but
       representing extrapolation beyond validated conditions at 75 bar.
       Reduce K₃ (see params dict) or pre-zero feed H₂O for differential comparisons.

    R4–R7 retain Bercic/Kopyscinski/Tronconi sub-models (same as KineticsGraaf).

    Reference:
      [VBF96]  Vanden Bussche & Froment, J. Catal. 161 (1996) 1-10
    """
    MODEL_NAME = "Vanden Bussche & Froment (1996)"
    MODEL_REF  = "Vanden Bussche & Froment, J. Catal. 161 (1996) 1-10"

    def __init__(self, params: dict = None):
        self.p = params if params is not None else self._default_params()

    @staticmethod
    def _default_params() -> dict:
        """
        Parameters from VBF (1996) Table 4 + R4-R7 Graaf sub-models.

        VBF rate constant units:
          k5*: mol/(kg_cat·s·bar²)  — r_MeOH numerator has f_CO₂·f_H₂ [bar²]
          k1*: mol/(kg_cat·s·bar)   — r_RWGS numerator has f_CO₂ [bar]
        """
        return {
            # ---- VBF-R1: CO₂ + 3H₂ ↔ MeOH + H₂O  [VBF 1996, Table 4] ----
            'A_meoh':     1.07,       # mol/(kg_cat·s·bar²)
            'Ea_meoh':    36696.,     # J/mol

            # ---- VBF-R2: CO₂ + H₂ ↔ CO + H₂O  (reverse WGS)  [VBF 1996, Table 4] ----
            'A_rwgs':     1.22e10,    # mol/(kg_cat·s·bar)
            'Ea_rwgs':    94765.,     # J/mol

            # ---- Denominator adsorption terms  [VBF 1996, Table 4] ----
            # K₃ = K_H₂O / (K₈·K₉·√K_H₂)  treated as temperature-independent constant
            # Fitted value at 210–260°C. Reduce for high-P (>50 bar) extrapolation.
            'K3':          3453.38,   # dimensionless   ← dominant H₂O/H₂ inhibitor
            'B_sqrtKH2':   0.499,     # bar^-0.5        √K_H₂ pre-exponential
            'dH_sqrtKH2':  17197.,    # J/mol           (endothermic surface step → K decreases with T)
            'B_KH2O':      6.62e-11,  # bar^-1          K_H₂O pre-exponential
            'dH_KH2O':     124119.,   # J/mol           (exothermic ads. → K decreases with rising T)

            # ---- R4: DME  [Bercic & Levec 1992] ----
            'A4':       3.7e2,        'Ea4':      80000.,
            'B_MeOH':   5.39e-5,     'dH_MeOH':  70400.,

            # ---- R5: Methanation  [Kopyscinski 2010] ----
            'A5':       6.1e-1,       'Ea5':      115000.,
            'B_CO_m':   1.2e-6,      'dH_CO_m':  70000.,

            # ---- R6: Ethanol  [Tronconi 1992] — uses Graaf ads. constants ----
            'A6':       2.5e-7,       'Ea6':      95000.,
            'B_CO':     7.99e-7,     'dH_CO':    58100.,
            'B_CO2':    1.02e-7,     'dH_CO2':   67400.,

            # ---- R7: 1-Propanol  [ASF / Bozzano & Manenti 2016] ----
            'alpha0':   0.10,         'E_alpha':  5000.,
        }

    def rate_constants(self, T: float) -> dict:
        """Evaluate all temperature-dependent rate and adsorption constants."""
        p = self.p
        k = {}
        k['k_meoh']   = p['A_meoh']    * np.exp(-p['Ea_meoh']    / (R_GAS * T))
        k['k_rwgs']   = p['A_rwgs']    * np.exp(-p['Ea_rwgs']    / (R_GAS * T))
        # VBF adsorption: √K_H₂ decreases with T (–sign); K_H₂O decreases with T (+sign)
        k['sqrtKH2']  = p['B_sqrtKH2'] * np.exp(-p['dH_sqrtKH2'] / (R_GAS * T))
        k['KH2O']     = p['B_KH2O']    * np.exp(+p['dH_KH2O']    / (R_GAS * T))
        k['K3']       = p['K3']        # treated as constant in validated T range
        # R4-R7 sub-models
        k['k4']       = p['A4']        * np.exp(-p['Ea4']         / (R_GAS * T))
        k['K_MeOH']   = p['B_MeOH']   * np.exp(+p['dH_MeOH']    / (R_GAS * T))
        k['k5']       = p['A5']        * np.exp(-p['Ea5']         / (R_GAS * T))
        k['K_CO_m']   = p['B_CO_m']   * np.exp(+p['dH_CO_m']     / (R_GAS * T))
        k['k6']       = p['A6']        * np.exp(-p['Ea6']         / (R_GAS * T))
        k['K_CO']     = p['B_CO']      * np.exp(+p['dH_CO']       / (R_GAS * T))
        k['K_CO2']    = p['B_CO2']     * np.exp(+p['dH_CO2']      / (R_GAS * T))
        k['alpha']    = p['alpha0']    * np.exp(-p['E_alpha']      / (R_GAS * T))
        return k

    def rates(self, T: float, f: np.ndarray) -> np.ndarray:
        """
        Compute 7 reaction rates [mol/(kg_cat·s)].

        r[0] = 0             (CO direct path absent in VBF)
        r[1] = r_MeOH        (CO₂ → MeOH via VBF-R1)
        r[2] = –r_RWGS       (sign flip: our R3 is WGS direction; VBF-R2 is RWGS)
        r[3]–r[6]            same Bercic / Kopyscinski / Tronconi sub-models
        """
        k   = self.rate_constants(T)
        r   = np.zeros(7)
        Keq = [ThermoModel.keq(i, T) for i in range(7)]

        f_CO   = max(f[0], 0.0)
        f_CO2  = max(f[1], 0.0)
        f_H2   = max(f[2], 1e-8)
        f_H2O  = max(f[3], 0.0)
        f_MeOH = max(f[4], 0.0)
        f_DME  = max(f[5], 0.0)
        f_EtOH = max(f[7], 0.0)

        # VBF common denominator
        den_VBF = (1.0
                   + k['K3']      * (f_H2O / f_H2)
                   + k['sqrtKH2'] * f_H2**0.5
                   + k['KH2O']    * f_H2O)
        den_VBF = max(den_VBF, 1e-30)

        # ---- r[0]: CO direct hydrogenation — zero in VBF ----
        r[0] = 0.0

        # ---- r[1]: CO₂ + 3H₂ ↔ MeOH + H₂O  [VBF-R1] ----
        # Keq[1] is K_eq for CO₂+3H₂↔MeOH+H₂O [bar⁻²] (our rxn idx=1)
        rev1 = f_H2O * f_MeOH / (Keq[1] * max(f_H2**3 * f_CO2, 1e-40))
        r[1] = k['k_meoh'] * f_CO2 * f_H2 * (1.0 - rev1) / den_VBF**3

        # ---- r[2]: WGS direction (our convention: CO+H₂O → CO₂+H₂ = forward) ----
        # VBF-R2 is RWGS (CO₂+H₂ → CO+H₂O); sign flip to match our stoichiometry
        # K_RWGS = 1/Keq[2];  r_RWGS > 0 when CO₂+H₂ has excess driving force
        rev2_rwgs = Keq[2] * f_CO * f_H2O / max(f_CO2 * f_H2, 1e-30)
        r_rwgs    = k['k_rwgs'] * f_CO2 * (1.0 - rev2_rwgs) / den_VBF
        r[2] = -r_rwgs   # positive r[2] = net WGS forward (CO→CO₂)

        # ---- r[3]: 2MeOH ↔ DME + H₂O  [Bercic & Levec 1992] ----
        revDME        = f_DME * f_H2O / (Keq[3] * max(f_MeOH**2, 1e-20))
        DrivingForce4 = k['K_MeOH']**2 * f_MeOH**2 * (1.0 - revDME)
        den4          = max((1.0 + 2.0*(k['K_MeOH']*f_MeOH)**0.5)**4, 1e-30)
        r[3] = k['k4'] * DrivingForce4 / den4

        # ---- r[4]: Methanation  CO + 3H₂ → CH₄ + H₂O  [Kopyscinski 2010] ----
        den5 = max((1.0 + k['K_CO_m']*f_CO)**2, 1e-30)
        r[4] = k['k5'] * f_CO * f_H2**3 / den5

        # ---- r[5]: Ethanol  [Tronconi 1992] ----
        den_ads  = (1.0 + k['K_CO']*f_CO + k['K_CO2']*f_CO2)
        revEtOH  = f_EtOH * f_H2O / (Keq[5] * max(f_CO**2 * f_H2**4, 1e-40))
        r[5] = (k['k6'] * k['K_CO']**2 * f_CO**2 * f_H2**2
                * (1.0 - revEtOH) / max(den_ads**3, 1e-30))

        # ---- r[6]: 1-Propanol  [ASF / Bozzano & Manenti 2016] ----
        r[6] = min(k['alpha'], 0.5) * abs(r[5])

        return r


# =============================================================================
# SECTION 4c: PARK et al. (2014) KINETICS
# =============================================================================

class KineticsPark(KineticsBase):
    """
    Park et al. (2014) kinetics for commercial Cu/ZnO/Al₂O₃ catalyst.

    Same 3-reaction Graaf LHHW functional form, but parameters re-fitted to
    experimental data from a commercial MegaMax-type catalyst at:
      T = 220–260 °C,  P = 20–50 bar,  GHSV = 4000–16000 h⁻¹

    Key differences vs KineticsGraaf (Nestler calibration):
      • Lower Ea for R2 (CO₂ path) — CO₂ hydrogenation is more T-sensitive
      • Higher R2 pre-exponential  — CO₂ dominates methanol formation
      • Slightly lower Ea for R3   — WGS reaches equilibrium faster
      • Smaller K_Phi adsorption   — different H₂O/H₂ balance on their catalyst

    Rate expressions (identical structure to KineticsGraaf):
      r₁ = k₁·K_CO ·(f_CO·f_H₂^1.5 – f_MeOH/(f_H₂^0.5·K_eq1)) / den
      r₂ = k₂·K_CO₂·(f_CO₂·f_H₂^1.5 – f_MeOH·f_H₂O/(f_H₂^1.5·K_eq2)) / den
      r₃ = k₃·K_CO₂·(f_CO₂·f_H₂ – f_CO·f_H₂O / K_eq3) / den
      den = (1 + K_CO·f_CO + K_CO₂·f_CO₂)·(f_H₂^0.5 + Φ·f_H₂O)

    ⚠  Parameters validated at 20–50 bar. Extrapolation to 75 bar should be
       confirmed against plant data using the .calibrate() interface.

    Reference:
      [PAR14]  Park et al., Fuel 118 (2014) 202-213
    """
    MODEL_NAME = "Park et al. (2014)"
    MODEL_REF  = "Park et al., Fuel 118 (2014) 202-213"

    def __init__(self, params: dict = None):
        self.p = params if params is not None else self._default_params()

    @staticmethod
    def _default_params() -> dict:
        """
        Kinetic parameters from Park et al. (2014) Table 4.
        Same functional form as Graaf (1988); different numerical values
        reflect fitting to a different commercial catalyst and conditions.

        Note: R4–R7 sub-model parameters (DME, methanation, alcohols) retain
        Bercic/Kopyscinski/Tronconi values — Park (2014) did not re-fit these.
        """
        return {
            # ---- R1: CO + 2H₂ ↔ MeOH  [Park 2014, Table 4] ----
            # Weaker pre-exp; CO path is secondary at industrial CO₂-rich feeds
            'A1':     0.0458,      # mol/(kg_cat·s·bar^1.5)
            'Ea1':    40000.,      # J/mol

            # ---- R2: CO₂ + 3H₂ ↔ MeOH + H₂O  [Park 2014, Table 4] ----
            # Dominant path; higher pre-exp + lower Ea relative to Graaf
            'A2':     81.00,       # mol/(kg_cat·s·bar^1.5)
            'Ea2':    11600.,      # J/mol

            # ---- R3: WGS  [Park 2014, Table 4] ----
            'A3':     2.25e8,      # mol/(kg_cat·s·bar^1.5)
            'Ea3':    109000.,     # J/mol

            # ---- Adsorption constants  [Park 2014, Table 4] ----
            'B_CO':    7.99e-7,   'dH_CO':    58100.,   # bar^-1, J/mol
            'B_CO2':   1.02e-7,   'dH_CO2':   67400.,
            'B_Phi':   2.07e-11,  'dH_Phi':   108100.,  # smaller Phi → less H₂O inhibition

            # ---- R4: DME  [Bercic & Levec 1992] (unchanged) ----
            'A4':      3.7e2,     'Ea4':       80000.,
            'B_MeOH':  5.39e-5,  'dH_MeOH':   70400.,

            # ---- R5: Methanation  [Kopyscinski 2010] (unchanged) ----
            'A5':      6.1e-1,    'Ea5':       115000.,
            'B_CO_m':  1.2e-6,   'dH_CO_m':   70000.,

            # ---- R6: Ethanol  [Tronconi 1992] (unchanged) ----
            'A6':      2.5e-7,    'Ea6':       95000.,

            # ---- R7: 1-Propanol  [ASF/Bozzano 2016] (unchanged) ----
            'alpha0':  0.10,      'E_alpha':   5000.,
        }

    def rate_constants(self, T: float) -> dict:
        """Evaluate all rate and adsorption constants — identical structure to KineticsGraaf."""
        p = self.p
        k = {}
        k['k1']     = p['A1']    * np.exp(-p['Ea1']    / (R_GAS * T))
        k['k2']     = p['A2']    * np.exp(-p['Ea2']    / (R_GAS * T))
        k['k3']     = p['A3']    * np.exp(-p['Ea3']    / (R_GAS * T))
        k['k4']     = p['A4']    * np.exp(-p['Ea4']    / (R_GAS * T))
        k['k5']     = p['A5']    * np.exp(-p['Ea5']    / (R_GAS * T))
        k['k6']     = p['A6']    * np.exp(-p['Ea6']    / (R_GAS * T))
        k['K_CO']   = p['B_CO']  * np.exp(+p['dH_CO']  / (R_GAS * T))
        k['K_CO2']  = p['B_CO2'] * np.exp(+p['dH_CO2'] / (R_GAS * T))
        k['Phi']    = p['B_Phi'] * np.exp(+p['dH_Phi'] / (R_GAS * T))
        k['K_MeOH'] = p['B_MeOH']* np.exp(+p['dH_MeOH']/ (R_GAS * T))
        k['K_CO_m'] = p['B_CO_m']* np.exp(+p['dH_CO_m'] / (R_GAS * T))
        k['alpha']  = p['alpha0'] * np.exp(-p['E_alpha'] / (R_GAS * T))
        return k

    def rates(self, T: float, f: np.ndarray) -> np.ndarray:
        """
        Compute 7 reaction rates [mol/(kg_cat·s)].
        Functional form identical to KineticsGraaf; only parameter values differ.
        """
        k   = self.rate_constants(T)
        r   = np.zeros(7)
        Keq = [ThermoModel.keq(i, T) for i in range(7)]

        f_CO   = max(f[0], 0.0)
        f_CO2  = max(f[1], 0.0)
        f_H2   = max(f[2], 1e-8)
        f_H2O  = max(f[3], 0.0)
        f_MeOH = max(f[4], 0.0)
        f_DME  = max(f[5], 0.0)
        f_EtOH = max(f[7], 0.0)

        # Graaf common denominator (same form as KineticsGraaf)
        den_ads = (1.0 + k['K_CO']*f_CO + k['K_CO2']*f_CO2)
        den_h   = (f_H2**0.5 + k['Phi']*f_H2O)
        den     = max(den_ads * den_h, 1e-30)

        # ---- R1: CO + 2H₂ ↔ MeOH  [Park 2014] ----
        r[0] = (k['k1'] * k['K_CO']
                * (f_CO*f_H2**1.5 - f_MeOH / (max(f_H2**0.5, 1e-8) * Keq[0]))
                / den)

        # ---- R2: CO₂ + 3H₂ ↔ MeOH + H₂O  [Park 2014 — dominant path] ----
        r[1] = (k['k2'] * k['K_CO2']
                * (f_CO2*f_H2**1.5 - f_MeOH*f_H2O / (max(f_H2**1.5, 1e-12) * Keq[1]))
                / den)

        # ---- R3: WGS  [Park 2014] ----
        r[2] = (k['k3'] * k['K_CO2']
                * (f_CO2*f_H2 - f_CO*f_H2O / Keq[2])
                / den)

        # ---- R4: DME  [Bercic & Levec 1992] ----
        revDME        = f_DME * f_H2O / (Keq[3] * max(f_MeOH**2, 1e-20))
        DrivingForce4 = k['K_MeOH']**2 * f_MeOH**2 * (1.0 - revDME)
        den4          = max((1.0 + 2.0*(k['K_MeOH']*f_MeOH)**0.5)**4, 1e-30)
        r[3] = k['k4'] * DrivingForce4 / den4

        # ---- R5: Methanation  [Kopyscinski 2010] ----
        den5 = max((1.0 + k['K_CO_m']*f_CO)**2, 1e-30)
        r[4] = k['k5'] * f_CO * f_H2**3 / den5

        # ---- R6: Ethanol  [Tronconi 1992] ----
        revEtOH = f_EtOH * f_H2O / (Keq[5] * max(f_CO**2 * f_H2**4, 1e-40))
        r[5] = (k['k6'] * k['K_CO']**2 * f_CO**2 * f_H2**2
                * (1.0 - revEtOH) / max(den_ads**3, 1e-30))

        # ---- R7: 1-Propanol  [ASF / Bozzano & Manenti 2016] ----
        r[6] = min(k['alpha'], 0.5) * abs(r[5])

        return r


# =============================================================================
# SECTION 4d: KINETICS FACTORY CLASS
# =============================================================================

class KineticsModel:
    """
    Factory wrapper — selects and instantiates the active kinetics model.

    Usage
    -----
    kin = KineticsModel('graaf')   # or 'vbf', 'park'
    kin.rates(T, f)                # returns 7-element rates array
    print(kin.MODEL_NAME)          # human-readable model label

    Alternatively pass a pre-built instance directly to DigitalTwin:
    kin = KineticsGraaf(my_params)
    twin = DigitalTwin(kinetics=kin)
    """
    REGISTRY = {
        'graaf': KineticsGraaf,
        'vbf':   KineticsVBF,
        'park':  KineticsPark,
    }

    def __init__(self, model: str = 'graaf', params: dict = None):
        model_key = model.lower().strip()
        if model_key not in self.REGISTRY:
            raise ValueError(
                f"Unknown kinetics model '{model}'. "
                f"Available: {list(self.REGISTRY.keys())}"
            )
        self._model = self.REGISTRY[model_key](params)
        self.model_key = model_key

    # Delegate core methods to the wrapped model
    def rates(self, T: float, f: np.ndarray) -> np.ndarray:
        return self._model.rates(T, f)

    def rate_constants(self, T: float) -> dict:
        return self._model.rate_constants(T)

    @property
    def MODEL_NAME(self) -> str:
        return self._model.MODEL_NAME

    @property
    def MODEL_REF(self) -> str:
        return self._model.MODEL_REF

    @property
    def p(self) -> dict:
        return self._model.p


# =============================================================================
# SECTION 5: HEAT TRANSFER MODEL
# =============================================================================

class HeatTransferModel:
    """
    Heat transfer coefficients for BWR multi-tubular reactor.

    References:
      - Li & Finlayson (1977) for tube-side Nu correlation
      - Gnielinski (1978) [Germany] for packed bed heat transfer
      - VDI Heat Atlas (2010) [Germany] for boiling HTC
      - Winterberg et al. (2000) [TU Munich/Magdeburg, Germany] for radial conduction
    """

    @staticmethod
    def htc_tube_side(T: float, P_bar: float, y: np.ndarray,
                      v_s: float, d_p: float, d_t: float) -> float:
        """
        Tube-side (gas-wall) heat transfer coefficient [W/(m^2·K)].
        Uses Gnielinski/Li-Finlayson correlation for packed beds.

        Parameters
        ----------
        v_s   : superficial gas velocity [m/s]
        d_p   : particle diameter [m]
        d_t   : tube inner diameter [m]
        """
        mu_mix  = TransportModel.viscosity_mix(y, T)
        lam_mix = TransportModel.conductivity_mix(y, T)
        cp_mix  = ThermoModel.cp_mix(y, T)
        rho     = P_bar * 1e5 * np.dot(y, MW) / (1000.0 * R_GAS * T)  # kg/m^3

        Re = rho * v_s * d_p / mu_mix
        Pr = mu_mix * cp_mix / lam_mix
        Re = max(Re, 1.0)
        Pr = max(Pr, 0.1)

        # Li & Finlayson (1977) + wall-effect correction
        Nu = 0.2 * Re**0.8 * Pr**0.33 * (1.0 + 2.0*d_p/d_t)**0.5
        h  = Nu * lam_mix / d_p
        return max(h, 10.0)

    @staticmethod
    def htc_shell_boiling(T_sat: float = 513.15,
                          q_flux: float = 50000.) -> float:
        """
        Shell-side boiling water heat transfer coefficient [W/(m^2·K)].
        Nucleate boiling on shell side — nearly constant T.
        Based on VDI Heat Atlas (2010), Stephan-Abdelsalam (1980).

        For BWR reactor, h_boiling is dominated by nucleate boiling.
        Typical range: 5000–15000 W/(m^2·K).
        Returns an approximate value; calibrate from plant data.
        """
        # Simplified: h ~ C * q_flux^0.67 at saturation
        C = 1.8e-2   # empirical constant for water nucleate boiling
        return max(C * q_flux**0.67, 5000.)

    @staticmethod
    def overall_htc(h_tube: float, h_shell: float = 8000.,
                    t_wall: float = 0.003, k_wall: float = 50.0,
                    r_inner: float = 0.019) -> float:
        """
        Overall heat transfer coefficient (tube-based) [W/(m^2·K)].
        1/U = 1/h_tube + t_wall/k_wall + 1/h_shell

        Parameters
        ----------
        t_wall : tube wall thickness [m] (typical: 2-4 mm)
        k_wall : wall thermal conductivity (steel) [W/(m·K)]
        r_inner: tube inner radius [m]
        """
        return 1.0 / (1.0/max(h_tube, 1.0) + t_wall/k_wall + 1.0/max(h_shell, 1.0))

    @staticmethod
    def effective_conductivity_bed(lam_f: float, lam_s: float,
                                    eps: float, v_s: float, d_p: float,
                                    rho_f: float, cp_f: float) -> float:
        """
        Effective radial thermal conductivity [W/(m·K)].
        Zehner-Schlunder model + convective contribution.

        Reference: Zehner & Schlunder (1970) [KIT, Germany]
        """
        B  = 1.25 * ((1.0 - eps)/eps)**(10./9.)
        lam_r = lam_s / lam_f
        term1 = 2.0 / (1.0 - B*lam_f/lam_s)
        kappa = lam_r
        ZS = (1.0 - np.sqrt(1.0 - eps)) + np.sqrt(1.0 - eps) * \
             (2.0 / (1.0 - B/kappa) * ((kappa-1.0)/(1.0-B/kappa)**2 * np.log(kappa/B) -
              (B+1.0)/2.0))
        lam_eff = ZS * lam_f + 0.1 * rho_f * cp_f * v_s * d_p
        return max(lam_eff, lam_f)


# =============================================================================
# SECTION 6: PENG-ROBINSON EOS — FUGACITY COEFFICIENTS
# =============================================================================

class PengRobinsonEOS:
    """
    Peng-Robinson EOS for fugacity coefficient calculation.

    Reference: Peng & Robinson (1976), Ind. Eng. Chem. Fundam. 15, 59-64.
    Applied to methanol synthesis: Graaf et al. (1990) [Netherlands].

    At 50-100 bar, 220-280°C: phi_i deviates 2-15% from ideal gas.
    """

    @staticmethod
    def fugacity_coefficients(y: np.ndarray, T: float, P_bar: float) -> np.ndarray:
        """
        Compute fugacity coefficients phi_i for all species.
        Uses van der Waals one-fluid mixing rules.

        Returns phi_i such that f_i = phi_i * y_i * P
        """
        P_Pa = P_bar * 1e5
        phi  = np.ones(N_SPECIES)   # default: ideal gas

        a_pure = np.zeros(N_SPECIES)
        b_pure = np.zeros(N_SPECIES)

        for i, sp in enumerate(SPECIES):
            Tc, Pc, omega = CRIT[sp]
            Pc_Pa = Pc * 1e5
            kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
            alpha = (1.0 + kappa*(1.0 - np.sqrt(T/Tc)))**2
            a_pure[i] = 0.45724 * R_GAS**2 * Tc**2 / Pc_Pa * alpha
            b_pure[i] = 0.07780 * R_GAS * Tc / Pc_Pa

        # Van der Waals mixing rules
        a_mix = 0.0
        b_mix = 0.0
        for i in range(N_SPECIES):
            b_mix += y[i] * b_pure[i]
            for j in range(N_SPECIES):
                kij = 0.0   # binary interaction parameter (default: 0)
                a_mix += y[i]*y[j]*np.sqrt(a_pure[i]*a_pure[j])*(1.0-kij)

        A_mix = a_mix * P_Pa / (R_GAS*T)**2
        B_mix = b_mix * P_Pa / (R_GAS*T)

        # Solve cubic: Z^3 - (1-B)*Z^2 + (A-3B^2-2B)*Z - (AB-B^2-B^3) = 0
        coeffs = [1.0, -(1.0-B_mix), (A_mix - 3.0*B_mix**2 - 2.0*B_mix),
                  -(A_mix*B_mix - B_mix**2 - B_mix**3)]
        roots = np.roots(coeffs)
        # Select largest real root (vapor phase)
        Z = max(z.real for z in roots if abs(z.imag) < 1e-6 and z.real > 0)

        # Fugacity coefficients
        for i in range(N_SPECIES):
            sum_ya = sum(y[j]*np.sqrt(a_pure[i]*a_pure[j]) for j in range(N_SPECIES))
            ln_phi = b_pure[i]/b_mix*(Z-1.0) - np.log(Z-B_mix) - \
                     A_mix/(2.0*np.sqrt(2.0)*B_mix) * \
                     (2.0*sum_ya/a_mix - b_pure[i]/b_mix) * \
                     np.log((Z+(1.0+np.sqrt(2.0))*B_mix)/(Z+(1.0-np.sqrt(2.0))*B_mix))
            phi[i] = np.exp(np.clip(ln_phi, -5.0, 5.0))

        return phi


# =============================================================================
# SECTION 7: EFFECTIVENESS FACTOR (THIELE MODULUS)
# =============================================================================

def effectiveness_factor(rates: np.ndarray, T: float, P_bar: float,
                          y: np.ndarray, rho_cat_bulk: float,
                          d_p: float, eps_p: float = 0.50,
                          tau: float = 3.5) -> np.ndarray:
    """
    Internal effectiveness factor for each reaction via Thiele modulus.

    Uses spherical approximation with first-order Thiele modulus.
    For multi-reaction system, use the rate-limiting species.

    Reference:
      Froment, Bischoff & De Wilde (2011), "Chemical Reactor Analysis and Design"
      3rd Ed., Wiley [G. Froment — Belgium/Netherlands]

    Parameters
    ----------
    rates       : array of 7 rates [mol/(kg_cat·s)]
    d_p         : particle diameter [m]
    eps_p       : intraparticle porosity [-]
    tau         : tortuosity factor [-] (typical: 3-4 for Cu/ZnO)

    Returns
    -------
    eta         : effectiveness factor array [0, 1] for each reaction
    """
    R_p  = d_p / 2.0   # particle radius [m]
    eta  = np.ones(7)

    # Limiting species index for each reaction (the reactant with smallest mole fraction)
    limiting = [0, 1, 2, 4, 0, 0, 0]  # CO, CO2, CO2, MeOH, CO, CO, CO

    for rxn in range(7):
        lim_sp = limiting[min(rxn, len(limiting)-1)]
        sp_name = SPECIES[lim_sp]
        C_surf  = y[lim_sp] * P_bar * 1e5 / (R_GAS * T)   # mol/m^3

        if C_surf < 1e-10 or abs(rates[rxn]) < 1e-20:
            eta[rxn] = 1.0
            continue

        D_eff = (eps_p / tau) * TransportModel.diffusivity_mix(sp_name, y, T, P_bar)
        rho_cat = rho_cat_bulk

        # First-order Thiele modulus (sphere)
        phi_sq = (R_p/3.0)**2 * rho_cat * abs(rates[rxn]) / (D_eff * max(C_surf, 1e-10))
        phi_n  = max(np.sqrt(phi_sq), 1e-8)

        # Analytical effectiveness factor for 1st-order sphere
        eta[rxn] = (1.0/phi_n) * (1.0/np.tanh(3.0*phi_n) - 1.0/(3.0*phi_n))
        eta[rxn] = np.clip(eta[rxn], 0.01, 1.0)

    return eta


# =============================================================================
# SECTION 8: REACTOR ODE SYSTEM
# =============================================================================

class ReactorODE:
    """
    Assembles and evaluates the ODE right-hand side for the reactor model.

    State vector X = [F_CO, F_CO2, F_H2, F_H2O, F_MeOH, F_DME, F_CH4,
                      F_EtOH, F_PrOH, F_N2,   P,   T]
                      0      1      2     3     4     5     6
                      7      8      9    10    11

    Governing equations:
      dFi/dz = sum_j(nu_ij * r_j * eta_j) * rho_cat * A_c   [mol/s/m]
      dP/dz  = Ergun pressure drop                           [Pa/m]
      dT/dz  = energy balance (0 for isothermal)             [K/m]
    """

    def __init__(self, reactor_params: dict, kinetics: KineticsModel,
                 thermo: ThermoModel, transport: TransportModel,
                 ht_model: HeatTransferModel,
                 isothermal: bool = True,
                 use_pr_eos: bool = True,
                 use_eta: bool = True):

        self.rp     = reactor_params
        self.kin    = kinetics
        self.thermo = thermo
        self.trans  = transport
        self.ht     = ht_model
        self.isothermal = isothermal
        self.use_pr_eos = use_pr_eos
        self.use_eta    = use_eta

    def __call__(self, z: float, X: np.ndarray) -> np.ndarray:
        """
        ODE right-hand side. Called by scipy integrator.

        Parameters
        ----------
        z : axial position [m]
        X : state vector (12 elements)

        Returns
        -------
        dX/dz : derivatives (12 elements)
        """
        rp = self.rp
        dXdz = np.zeros(12)

        # Unpack state
        F     = np.maximum(X[:10], 0.0)    # molar flows [mol/s]
        P_Pa  = max(X[10], 1e4)            # pressure [Pa]
        T     = max(X[11], 150.0 + 273.15) # temperature [K]
        P_bar = P_Pa / 1e5

        F_tot = max(np.sum(F), 1e-12)
        y     = F / F_tot                   # mole fractions

        # Mixture molecular weight [kg/mol]
        M_mix = np.dot(y, MW) / 1000.0

        # Gas density [kg/m^3]
        rho_g = P_Pa * M_mix / (R_GAS * T)

        # Superficial velocity [m/s]
        Q_vol = F_tot * R_GAS * T / P_Pa    # volumetric flow [m^3/s]
        v_s   = Q_vol / rp['A_c']

        # Fugacity coefficients (PR-EOS or ideal)
        if self.use_pr_eos:
            phi = PengRobinsonEOS.fugacity_coefficients(y, T, P_bar)
        else:
            phi = np.ones(N_SPECIES)

        # Fugacities [bar]
        f = phi * y * P_bar

        # Reaction rates [mol/(kg_cat·s)]
        rates_raw = self.kin.rates(T, f)

        # Internal effectiveness factors
        if self.use_eta:
            eta = effectiveness_factor(
                rates_raw, T, P_bar, y,
                rp['rho_bulk'], rp['d_p'])
        else:
            eta = np.ones(7)

        rates = rates_raw * eta   # effective rates

        # ---- Species balances (dFi/dz) ----
        # dFi/dz = sum_j(nu_ij * r_j) * rho_cat * A_c  [mol/(s·m)]
        rho_cat_vol = rp['rho_bulk']   # kg_cat/m^3_bed  (bulk density)
        for i in range(N_SPECIES):
            dXdz[i] = np.dot(STOICH[i, :], rates) * rho_cat_vol * rp['A_c']

        # ---- Pressure drop (Ergun equation) ----
        # dP/dz = -[150*(1-eps)^2 * mu * v / (dp^2 * eps^3)]
        #         -[1.75*(1-eps) * rho * v^2 / (dp * eps^3)]
        mu_mix = TransportModel.viscosity_mix(y, T)
        eps    = rp['eps']
        d_p    = rp['d_p']
        dPdz_viscous  = 150.0 * (1.0-eps)**2 * mu_mix * v_s / (d_p**2 * eps**3)
        dPdz_inertial = 1.75 * (1.0-eps) * rho_g * v_s**2 / (d_p * eps**3)
        dXdz[10] = -(dPdz_viscous + dPdz_inertial)

        # ---- Energy balance ----
        if self.isothermal:
            dXdz[11] = 0.0
        else:
            # dT/dz = [-sum(dHr_j * r_j) * rho_cat + U*pi*dt*(Ts-T)] / (Ftot * Cp_mix)
            Cp_mix  = ThermoModel.cp_mix(y, T)    # J/(mol·K)
            dHr_sum = sum(ThermoModel.dHr_T(j, T) * rates[j] for j in range(7))
            # Heat removal: U * pi * d_t * (T_shell - T)  [W/m]
            h_tube  = HeatTransferModel.htc_tube_side(T, P_bar, y, v_s, d_p, rp['d_t'])
            U_eff   = HeatTransferModel.overall_htc(h_tube, h_shell=rp.get('h_shell', 8000.))
            Q_cool  = U_eff * np.pi * rp['d_t'] * (rp['T_shell'] - T)   # W/m
            # Energy balance
            dXdz[11] = (-dHr_sum * rho_cat_vol * rp['A_c'] + Q_cool) / (F_tot * Cp_mix)

        return dXdz


# =============================================================================
# SECTION 9: DIGITAL TWIN CLASS
# =============================================================================

class DigitalTwin:
    """
    Top-level digital twin for the isothermal multi-tubular methanol reactor.

    Encapsulates:
      - Reactor geometry and operating conditions
      - Sub-models (kinetics, thermo, transport, heat transfer)
      - ODE integration (BDF solver for stiff system)
      - Post-processing and KPI calculation
      - Parameter calibration interface
    """

    # ---- Default industrial reactor parameters (Lurgi MRP / Air Liquide type) ----
    DEFAULT_REACTOR = {
        'd_t':       0.038,      # tube inner diameter [m] (38 mm)
        'L':         7.0,        # tube length [m]
        'N_tubes':   5000,       # number of tubes
        't_wall':    0.003,      # tube wall thickness [m]
        'k_wall':    50.0,       # wall thermal conductivity [W/(m·K)] (steel)
        'd_p':       0.006,      # catalyst particle diameter [m] (6 mm equiv. sphere)
        'eps':       0.40,       # bed void fraction [-]
        'rho_bulk':  1200.0,     # catalyst bulk density [kg/m^3]
        'T_shell':   533.15,     # shell-side (coolant) temperature [K] = 260°C
        'h_shell':   8000.0,     # shell-side HTC [W/(m^2·K)]
    }

    DEFAULT_FEED = {
        'T_in':      533.15,    # inlet temperature [K] = 260°C
        'P_in':      75e5,      # inlet pressure [Pa] = 75 bar
        'F_CO':      0.250,     # inlet CO molar flow per tube [mol/s]
        'F_CO2':     0.060,     # inlet CO2 molar flow per tube [mol/s]
        'F_H2':      0.620,     # inlet H2 molar flow per tube [mol/s]
        'F_N2':      0.040,     # inlet N2 molar flow per tube [mol/s]
        'F_H2O':     0.005,     # trace water
        'F_MeOH':    0.001,     # trace methanol
    }

    def __init__(self, reactor_params: dict = None, feed: dict = None,
                 kinetics: KineticsBase = None,
                 kinetics_model: str = 'graaf',
                 isothermal: bool = True,
                 use_pr_eos: bool = True,
                 use_eta: bool = True):
        """
        Parameters
        ----------
        reactor_params   : dict — geometry & operating params (see DEFAULT_REACTOR)
        feed             : dict — inlet conditions (see DEFAULT_FEED)
        kinetics         : KineticsBase instance — pass to override kinetics_model
        kinetics_model   : str — 'graaf' | 'vbf' | 'park'  (default: 'graaf')
        isothermal       : bool — True = isothermal mode (BWR boiling water)
        use_pr_eos       : bool — True = Peng-Robinson fugacities
        use_eta          : bool — True = internal effectiveness factors
        """
        self.rp  = {**self.DEFAULT_REACTOR, **(reactor_params or {})}
        self.fd  = {**self.DEFAULT_FEED,    **(feed or {})}
        # Kinetics: explicit instance takes precedence over model string
        if kinetics is not None:
            self.kin = kinetics
        else:
            self.kin = KineticsModel(kinetics_model)
        self.thermo    = ThermoModel()
        self.transport = TransportModel()
        self.ht_model  = HeatTransferModel()
        self.isothermal = isothermal
        self.use_pr_eos = use_pr_eos
        self.use_eta    = use_eta

        # Derived geometry
        self.rp['A_c'] = np.pi * self.rp['d_t']**2 / 4.0   # tube cross-section [m^2]
        self.rp['A_ht'] = np.pi * self.rp['d_t'] * self.rp['L']  # heat transfer area/tube [m^2]

        self.solution  = None   # stores last solve_ivp result
        self.z_grid    = None
        self.profiles  = None

    def initial_conditions(self) -> np.ndarray:
        """
        Build initial state vector X0 from feed conditions.
        X = [F_CO, F_CO2, F_H2, F_H2O, F_MeOH, F_DME, F_CH4, F_EtOH, F_PrOH, F_N2, P, T]
        """
        fd = self.fd
        X0 = np.array([
            fd.get('F_CO',   0.25),
            fd.get('F_CO2',  0.06),
            fd.get('F_H2',   0.62),
            fd.get('F_H2O',  0.005),
            fd.get('F_MeOH', 0.001),
            fd.get('F_DME',  0.0),
            fd.get('F_CH4',  0.0),
            fd.get('F_EtOH', 0.0),
            fd.get('F_PrOH', 0.0),
            fd.get('F_N2',   0.04),
            fd.get('P_in',   75e5),
            fd.get('T_in',   533.15),
        ], dtype=float)
        return X0

    def solve(self, n_points: int = 500) -> dict:
        """
        Integrate reactor ODE from z=0 to z=L.

        Uses scipy BDF solver (appropriate for stiff ODE systems arising
        from fast WGS + slow alcohol chain growth reactions).

        Reference:
          Stiff ODE discussion: Manenti et al. (2013) [PoliMi, Italy]
          Buzzi-Ferraris & Manenti (2009) [PoliMi, Italy]

        Returns
        -------
        profiles : dict with axial profiles of all state variables + KPIs
        """
        ode_rhs = ReactorODE(
            self.rp, self.kin, self.thermo, self.transport, self.ht_model,
            isothermal=self.isothermal,
            use_pr_eos=self.use_pr_eos,
            use_eta=self.use_eta
        )

        X0 = self.initial_conditions()
        z_span = (0.0, self.rp['L'])
        z_eval = np.linspace(0.0, self.rp['L'], n_points)

        sol = solve_ivp(
            ode_rhs,
            z_span,
            X0,
            method='BDF',           # stiff solver — appropriate for fast WGS + slow alcohols
            t_eval=z_eval,
            rtol=1e-5,
            atol=1e-7,
            dense_output=True,
            max_step=self.rp['L']/200.
        )

        if not sol.success:
            print(f"[WARNING] ODE solver: {sol.message}")

        self.solution = sol
        self.z_grid   = sol.t
        self.profiles = self._post_process(sol)
        return self.profiles

    def _post_process(self, sol) -> dict:
        """
        Post-process ODE solution: compute mole fractions, conversions,
        selectivities, space-time yield, pressure drop, etc.
        """
        z    = sol.t
        X    = sol.y
        F    = np.maximum(X[:10, :], 0.0)
        P    = X[10, :]
        T    = X[11, :]

        F_tot = np.sum(F, axis=0)
        y_mol = F / F_tot[np.newaxis, :]   # mole fractions

        # Feed values
        F0 = self.initial_conditions()
        F_CO0   = F0[0]
        F_CO20  = F0[1]
        F_H20   = F0[2]
        F_MeOH0 = F0[4]

        # Conversions
        X_CO    = (F_CO0  - F[0, :]) / max(F_CO0,  1e-12)
        X_CO2   = (F_CO20 - F[1, :]) / max(F_CO20, 1e-12)
        X_CO_tot= (F_CO0 + F_CO20 - F[0,:] - F[1,:]) / max(F_CO0+F_CO20, 1e-12)

        # Selectivities to MeOH (carbon-based)
        dF_CO   = F_CO0  - F[0, :]
        dF_CO2  = F_CO20 - F[1, :]
        dF_C    = dF_CO + dF_CO2
        C_to_MeOH = F[4, :] - F_MeOH0
        C_to_DME  = 2.0 * F[5, :]
        C_to_CH4  = F[6, :]
        C_to_EtOH = 2.0 * F[7, :]
        C_to_PrOH = 3.0 * F[8, :]

        denom_sel = np.maximum(dF_C, 1e-12)
        S_MeOH = np.clip(C_to_MeOH / denom_sel, 0, 1)
        S_DME  = np.clip(C_to_DME  / denom_sel, 0, 1)
        S_CH4  = np.clip(C_to_CH4  / denom_sel, 0, 1)
        S_EtOH = np.clip(C_to_EtOH / denom_sel, 0, 1)
        S_PrOH = np.clip(C_to_PrOH / denom_sel, 0, 1)

        # Space-time yield [kg_MeOH / (kg_cat * h)]
        V_bed   = self.rp['A_c'] * self.rp['L']
        W_cat   = V_bed * self.rp['rho_bulk']
        STY_MeOH = (F[4, :] - F_MeOH0) * MW[4] / 1000.0 * 3600.0 / W_cat  # kg/(kg_cat·h)

        # Productivity per reactor [kg MeOH / h]
        Q_MeOH_tube     = (F[4, -1] - F_MeOH0) * MW[4] / 1000.0 * 3600.0   # kg/h/tube
        Q_MeOH_reactor  = Q_MeOH_tube * self.rp['N_tubes']

        # Hydrogen:CO ratio
        H2_CO_ratio = F[2, :] / np.maximum(F[0, :] + F[1, :], 1e-12)

        # Atom balance check (C, H, O)
        C_in  = F_CO0 + F_CO20
        C_out = F[0,-1] + F[1,-1] + F[4,-1] + 2*F[5,-1] + F[6,-1] + 2*F[7,-1] + 3*F[8,-1]
        H_in  = 2*F0[2] + 2*F0[3] + 4*F_MeOH0
        H_out = (2*F[2,-1] + 2*F[3,-1] + 4*F[4,-1] +
                 6*F[5,-1] + 4*F[6,-1] + 6*F[7,-1] + 8*F[8,-1])
        C_err = abs(C_in - C_out) / max(C_in, 1e-12) * 100.
        H_err = abs(H_in - H_out) / max(H_in, 1e-12) * 100.

        return {
            'z':          z,
            'T':          T,
            'P_bar':      P / 1e5,
            'F':          F,
            'y':          y_mol,
            'X_CO':       X_CO,
            'X_CO2':      X_CO2,
            'X_CO_total': X_CO_tot,
            'S_MeOH':     S_MeOH,
            'S_DME':      S_DME,
            'S_CH4':      S_CH4,
            'S_EtOH':     S_EtOH,
            'S_PrOH':     S_PrOH,
            'STY_MeOH':   STY_MeOH,
            'H2_CO_ratio':H2_CO_ratio,
            'Q_MeOH_tube':    Q_MeOH_tube,
            'Q_MeOH_reactor': Q_MeOH_reactor,
            'C_balance_err%': C_err,
            'H_balance_err%': H_err,
            'W_cat_kg':   W_cat,
            '_kinetics_name': getattr(self.kin, 'MODEL_NAME', 'LHHW Kinetics'),
        }

    def print_summary(self):
        """Print key performance indicators at reactor outlet."""
        if self.profiles is None:
            print("Run .solve() first.")
            return
        pr = self.profiles
        kin_name = getattr(self.kin, 'MODEL_NAME', str(type(self.kin).__name__))
        print("\n" + "="*65)
        print("   METHANOL REACTOR DIGITAL TWIN — OUTLET SUMMARY")
        print("="*65)
        print(f"  Kinetics model          : {kin_name}")
        print(f"  Reactor length          : {self.rp['L']:.1f} m")
        print(f"  Tube diameter           : {self.rp['d_t']*1000:.1f} mm")
        print(f"  N_tubes                 : {self.rp['N_tubes']:,}")
        print(f"  Shell temperature       : {self.rp['T_shell']-273.15:.1f} °C")
        print(f"  Catalyst bulk density   : {self.rp['rho_bulk']:.0f} kg/m³")
        print(f"  Catalyst weight/tube    : {pr['W_cat_kg']:.2f} kg")
        print("-"*65)
        print(f"  Outlet T                : {pr['T'][-1]-273.15:.2f} °C")
        print(f"  Outlet P                : {pr['P_bar'][-1]:.2f} bar")
        print(f"  Pressure drop           : {pr['P_bar'][0]-pr['P_bar'][-1]:.3f} bar")
        print("-"*65)
        print(f"  CO conversion           : {pr['X_CO'][-1]*100:.2f} %")
        print(f"  CO₂ conversion          : {pr['X_CO2'][-1]*100:.2f} %")
        print(f"  Total C conversion      : {pr['X_CO_total'][-1]*100:.2f} %")
        print("-"*65)
        print(f"  C-selectivity MeOH      : {pr['S_MeOH'][-1]*100:.3f} %")
        print(f"  C-selectivity DME       : {pr['S_DME'][-1]*100:.4f} %")
        print(f"  C-selectivity CH₄       : {pr['S_CH4'][-1]*100:.5f} %")
        print(f"  C-selectivity EtOH      : {pr['S_EtOH'][-1]*100:.5f} %")
        print(f"  C-selectivity PrOH      : {pr['S_PrOH'][-1]*100:.6f} %")
        print("-"*65)
        print(f"  STY MeOH (outlet)       : {pr['STY_MeOH'][-1]:.4f} kg/(kg_cat·h)")
        print(f"  MeOH production/tube    : {pr['Q_MeOH_tube']:.4f} kg/h")
        print(f"  MeOH production/reactor : {pr['Q_MeOH_reactor']:.1f} kg/h")
        print(f"  MeOH production/reactor : {pr['Q_MeOH_reactor']/1000.:.2f} t/h")
        print("-"*65)
        print(f"  Carbon balance error    : {pr['C_balance_err%']:.4f} %")
        print(f"  Hydrogen balance error  : {pr['H_balance_err%']:.4f} %")
        print("="*65)

    def calibrate(self, exp_data: dict, param_names: list,
                  method: str = 'Nelder-Mead') -> dict:
        """
        Calibrate kinetic parameters against experimental data.

        Parameters
        ----------
        exp_data    : dict with measured values
                      {'X_CO': float, 'S_MeOH': float, 'X_CO2': float, ...}
        param_names : list of parameter names to optimize (keys in kinetics.p)
        method      : scipy.optimize method string

        Returns
        -------
        opt_params  : optimized parameter dictionary
        """
        p0 = [self.kin.p[name] for name in param_names]
        bounds = [(p*0.01, p*100.) for p in p0]

        def objective(log_p):
            params = dict(self.kin.p)
            for name, lp in zip(param_names, log_p):
                params[name] = np.exp(lp)
            self.kin.p = params
            try:
                pr = self.solve()
                loss = 0.0
                if 'X_CO' in exp_data:
                    loss += (pr['X_CO'][-1] - exp_data['X_CO'])**2 * 1e4
                if 'S_MeOH' in exp_data:
                    loss += (pr['S_MeOH'][-1] - exp_data['S_MeOH'])**2 * 1e4
                if 'X_CO2' in exp_data:
                    loss += (pr['X_CO2'][-1] - exp_data['X_CO2'])**2 * 1e4
                return loss
            except Exception:
                return 1e10

        log_p0 = np.log(p0)
        res = minimize(objective, log_p0, method=method,
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        opt_params = {n: np.exp(v) for n, v in zip(param_names, res.x)}
        self.kin.p.update(opt_params)
        print(f"Calibration converged: {res.success}, fun={res.fun:.6f}")
        return opt_params


# =============================================================================
# SECTION 10: VISUALIZATION
# =============================================================================

def plot_reactor_profiles(profiles: dict, save_fig: bool = False,
                           filename: str = 'reactor_profiles.png'):
    """
    Comprehensive multi-panel visualization of reactor profiles.

    Panels:
      1. Molar flow profiles (all species)
      2. Mole fraction profiles (key species)
      3. Conversion and selectivity vs z
      4. Temperature and pressure profiles
      5. Space-time yield and byproduct selectivities
      6. H2/CO ratio vs z
    """
    pr = profiles
    z  = pr['z']

    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor('#0f1117')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.38)

    col_main    = '#00d4ff'
    col_accent  = '#ff6b6b'
    col_green   = '#00ff88'
    col_yellow  = '#ffd700'
    col_purple  = '#cc88ff'
    col_orange  = '#ff9944'
    col_pink    = '#ff66bb'
    bg_ax       = '#1a1f2e'
    grid_col    = '#2a3040'

    SPECIES_COLORS = {
        'CO': col_accent, 'CO2': col_orange, 'H2': col_main,
        'H2O': col_purple, 'MeOH': col_green, 'DME': col_yellow,
        'CH4': col_pink, 'EtOH': '#88ff44', 'PrOH': '#44ffcc', 'N2': '#888888'
    }

    def style_ax(ax, title):
        ax.set_facecolor(bg_ax)
        ax.spines[:].set_color(grid_col)
        ax.tick_params(colors='#aabbcc', labelsize=9)
        ax.xaxis.label.set_color('#aabbcc')
        ax.yaxis.label.set_color('#aabbcc')
        ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=8)
        ax.grid(True, color=grid_col, linewidth=0.6, alpha=0.7)

    # --- Panel 1: Molar flow rates ---
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'Molar Flow Rates')
    for i, sp in enumerate(SPECIES):
        ax1.plot(z, pr['F'][i, :]*1000., label=sp,
                 color=SPECIES_COLORS[sp], linewidth=1.8)
    ax1.set_xlabel('Axial Position z [m]')
    ax1.set_ylabel('Molar Flow [mmol/s]')
    ax1.legend(fontsize=7, ncol=2, facecolor='#1a2030', labelcolor='white',
               edgecolor=grid_col, loc='center right')

    # --- Panel 2: Mole fractions ---
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'Gas-Phase Mole Fractions')
    show_sp = ['CO', 'CO2', 'H2', 'H2O', 'MeOH', 'N2']
    for sp in show_sp:
        i = SPECIES.index(sp)
        ax2.plot(z, pr['y'][i, :]*100., label=sp,
                 color=SPECIES_COLORS[sp], linewidth=1.8)
    ax2.set_xlabel('Axial Position z [m]')
    ax2.set_ylabel('Mole Fraction [mol%]')
    ax2.legend(fontsize=8, facecolor='#1a2030', labelcolor='white', edgecolor=grid_col)

    # --- Panel 3: Conversions ---
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, 'Conversions')
    ax3.plot(z, pr['X_CO']*100.,      label='X_CO',  color=col_accent, lw=2.2)
    ax3.plot(z, pr['X_CO2']*100.,     label='X_CO₂', color=col_orange, lw=2.2)
    ax3.plot(z, pr['X_CO_total']*100.,label='X_C (total)', color=col_green, lw=2.2, ls='--')
    ax3.set_xlabel('Axial Position z [m]')
    ax3.set_ylabel('Conversion [%]')
    ax3.legend(fontsize=9, facecolor='#1a2030', labelcolor='white', edgecolor=grid_col)
    ax3.set_ylim(0, 100)

    # --- Panel 4: Selectivities ---
    ax4 = fig.add_subplot(gs[1, 0])
    style_ax(ax4, 'Carbon Selectivities')
    ax4.plot(z, pr['S_MeOH']*100., label='S_MeOH', color=col_green, lw=2.2)
    ax4.plot(z, pr['S_DME']*100.,  label='S_DME',  color=col_yellow, lw=1.8)
    ax4.plot(z, pr['S_CH4']*100.,  label='S_CH₄',  color=col_pink, lw=1.8)
    ax4.plot(z, pr['S_EtOH']*100., label='S_EtOH', color='#88ff44', lw=1.5)
    ax4.plot(z, pr['S_PrOH']*100., label='S_PrOH', color='#44ffcc', lw=1.5)
    ax4.set_xlabel('Axial Position z [m]')
    ax4.set_ylabel('C-Selectivity [%]')
    ax4.legend(fontsize=8, facecolor='#1a2030', labelcolor='white', edgecolor=grid_col)

    # --- Panel 5: Temperature profile ---
    ax5 = fig.add_subplot(gs[1, 1])
    style_ax(ax5, 'Temperature Profile')
    ax5.plot(z, pr['T'] - 273.15, color=col_accent, lw=2.5, label='T_tube')
    T_shell = 260.  # default
    ax5.axhline(T_shell, color='#aaaaaa', ls='--', lw=1.5, label='T_shell')
    ax5.set_xlabel('Axial Position z [m]')
    ax5.set_ylabel('Temperature [°C]')
    ax5.legend(fontsize=9, facecolor='#1a2030', labelcolor='white', edgecolor=grid_col)
    dT_max = max(pr['T']) - min(pr['T'])
    ax5.text(0.05, 0.95, f'ΔT_max = {dT_max:.2f} K',
             transform=ax5.transAxes, color=col_yellow, fontsize=9, va='top')

    # --- Panel 6: Pressure profile ---
    ax6 = fig.add_subplot(gs[1, 2])
    style_ax(ax6, 'Pressure Drop (Ergun)')
    ax6.plot(z, pr['P_bar'], color=col_main, lw=2.2)
    ax6.set_xlabel('Axial Position z [m]')
    ax6.set_ylabel('Pressure [bar]')
    dP = pr['P_bar'][0] - pr['P_bar'][-1]
    ax6.text(0.05, 0.08, f'ΔP = {dP:.3f} bar',
             transform=ax6.transAxes, color=col_yellow, fontsize=10, fontweight='bold')

    # --- Panel 7: Space-time yield ---
    ax7 = fig.add_subplot(gs[2, 0])
    style_ax(ax7, 'Space-Time Yield — MeOH')
    ax7.plot(z, pr['STY_MeOH'], color=col_green, lw=2.2)
    ax7.set_xlabel('Axial Position z [m]')
    ax7.set_ylabel('STY [kg_MeOH/(kg_cat·h)]')
    ax7.fill_between(z, 0, pr['STY_MeOH'], alpha=0.2, color=col_green)

    # --- Panel 8: H2/CO ratio ---
    ax8 = fig.add_subplot(gs[2, 1])
    style_ax(ax8, 'H₂/CO Ratio (stoichiometric = 2)')
    ax8.plot(z, pr['H2_CO_ratio'], color=col_main, lw=2.2)
    ax8.axhline(2.0, color=col_yellow, ls='--', lw=1.5, label='Stoichiometric (R=2)')
    ax8.axhline(3.0, color=col_orange, ls='--', lw=1.0, label='R=3 (CO₂ path)')
    ax8.set_xlabel('Axial Position z [m]')
    ax8.set_ylabel('H₂/(CO+CO₂) ratio')
    ax8.legend(fontsize=8, facecolor='#1a2030', labelcolor='white', edgecolor=grid_col)

    # --- Panel 9: Byproduct mole fractions (zoomed) ---
    ax9 = fig.add_subplot(gs[2, 2])
    style_ax(ax9, 'Byproduct Profiles (ppm scale)')
    byp = ['DME', 'CH4', 'EtOH', 'PrOH']
    for sp in byp:
        i = SPECIES.index(sp)
        ax9.plot(z, pr['y'][i, :]*1e6, label=sp,
                 color=SPECIES_COLORS[sp], lw=1.8)
    ax9.set_xlabel('Axial Position z [m]')
    ax9.set_ylabel('Mole Fraction [ppm_mol]')
    ax9.legend(fontsize=9, facecolor='#1a2030', labelcolor='white', edgecolor=grid_col)

    # Title
    kin_name = getattr(profiles.get('_kinetics_name', None), '__str__', lambda: 'LHHW')()
    kin_label = profiles.get('_kinetics_name', 'Graaf + Bercic + Tronconi Kinetics')
    fig.text(0.5, 0.98,
             'METHANOL SYNTHESIS REACTOR — DIGITAL TWIN PROFILES',
             ha='center', va='top', fontsize=14, fontweight='bold', color='white')
    fig.text(0.5, 0.963,
             f'MegaMax Cu/ZnO/Al₂O₃ | 7-Reaction LHHW Network | {kin_label}',
             ha='center', va='top', fontsize=9, color='#88aabb')

    if save_fig:
        plt.savefig(filename, dpi=150, bbox_inches='tight',
                    facecolor='#0f1117', edgecolor='none')
        print(f"Figure saved: {filename}")
    return fig


def sensitivity_analysis(twin: DigitalTwin, n_samples: int = 20):
    """
    One-at-a-time sensitivity analysis for key operating parameters.

    Varies: T_shell (±20K), P_in (±15 bar), H2/CO ratio (±0.5), GHSV (±25%)
    Reports impact on X_CO, S_MeOH, STY_MeOH.
    """
    print("\n" + "="*60)
    print("  SENSITIVITY ANALYSIS — One-at-a-Time")
    print("="*60)

    base_fd  = dict(twin.fd)
    base_rp  = dict(twin.rp)

    def run_case(rp_update=None, fd_update=None):
        twin.rp = {**base_rp,  **(rp_update or {})}
        twin.fd = {**base_fd,  **(fd_update or {})}
        pr = twin.solve()
        return pr['X_CO'][-1], pr['S_MeOH'][-1], pr['STY_MeOH'][-1]

    # Base case
    X_CO_b, S_b, STY_b = run_case()
    print(f"\n{'Parameter':<25} {'Value':>10} {'X_CO[%]':>10} {'S_MeOH[%]':>12} {'STY[kg/kgc/h]':>15}")
    print("-"*72)
    print(f"{'Base case':<25} {'—':>10} {X_CO_b*100:>10.2f} {S_b*100:>12.3f} {STY_b:>15.5f}")

    # T_shell sweep
    for dT in [-20, -10, +10, +20]:
        T_new = base_rp['T_shell'] + dT
        X, S, STY = run_case(rp_update={'T_shell': T_new, 'A_c': base_rp['A_c']})
        print(f"{'T_shell ' + str(int(T_new-273.15)) + '°C':<25} {T_new-273.15:>10.0f} "
              f"{X*100:>10.2f} {S*100:>12.3f} {STY:>15.5f}")

    # Pressure sweep
    for dP_bar in [-15, -5, +5, +15]:
        P_new = base_fd['P_in'] + dP_bar*1e5
        X, S, STY = run_case(fd_update={'P_in': P_new})
        print(f"{'P_in ' + str(int(P_new/1e5)) + ' bar':<25} {P_new/1e5:>10.0f} "
              f"{X*100:>10.2f} {S*100:>12.3f} {STY:>15.5f}")

    # Restore
    twin.rp = base_rp
    twin.fd = base_fd
    print("="*60)


def compare_kinetics_models(reactor_params: dict, feed: dict,
                             models: list = None,
                             n_points: int = 300) -> dict:
    """
    Run the digital twin with multiple kinetics models and print a comparison table.

    Parameters
    ----------
    reactor_params : dict  — reactor geometry/operating parameters
    feed           : dict  — inlet conditions
    models         : list  — model keys to compare (default: all three)
    n_points       : int   — ODE integration points per run

    Returns
    -------
    results : dict  — {model_key: profiles_dict}  for further analysis

    Example
    -------
    results = compare_kinetics_models(reactor_params, feed, models=['graaf', 'park', 'vbf'])
    """
    if models is None:
        models = ['graaf', 'park', 'vbf']

    results = {}
    print("\n" + "="*75)
    print("  KINETICS MODEL COMPARISON")
    print("="*75)
    hdr = (f"{'Model':<38} {'X_CO[%]':>8} {'X_CO2[%]':>9} "
           f"{'S_MeOH[%]':>10} {'STY[kg/kgc/h]':>14}")
    print(hdr)
    print("-"*75)

    for key in models:
        try:
            twin = DigitalTwin(
                reactor_params = reactor_params,
                feed           = feed,
                kinetics_model = key,
                isothermal     = True,
                use_pr_eos     = True,
                use_eta        = True,
            )
            pr = twin.solve(n_points=n_points)
            results[key] = pr

            name = twin.kin.MODEL_NAME
            X_CO  = pr['X_CO'][-1]  * 100.
            X_CO2 = pr['X_CO2'][-1] * 100.
            S     = pr['S_MeOH'][-1] * 100.
            STY   = pr['STY_MeOH'][-1]
            print(f"  {name:<36} {X_CO:>8.2f} {X_CO2:>9.2f} {S:>10.3f} {STY:>14.5f}")

        except Exception as exc:
            print(f"  [{key}] ERROR: {exc}")

    print("="*75)
    return results


# =============================================================================
# SECTION 11: INTERACTIVE USER INPUT & MAIN EXECUTION
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# INPUT HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _ask_float(prompt_text: str, default: float,
               lo: float = None, hi: float = None) -> float:
    """
    Prompt user for a float value.
    Shows default in brackets; pressing Enter accepts it.
    Validates optional lower (lo) and upper (hi) bounds.
    """
    while True:
        raw = input(f"    {prompt_text} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
        except ValueError:
            print(f"      ✗  Please enter a valid number (or press Enter for default).")
            continue
        if lo is not None and val < lo:
            print(f"      ✗  Value must be ≥ {lo}.")
            continue
        if hi is not None and val > hi:
            print(f"      ✗  Value must be ≤ {hi}.")
            continue
        return val


def _ask_int(prompt_text: str, default: int,
             lo: int = None, hi: int = None) -> int:
    """Prompt user for an integer value."""
    while True:
        raw = input(f"    {prompt_text} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
        except ValueError:
            print(f"      ✗  Please enter a whole number (or press Enter for default).")
            continue
        if lo is not None and val < lo:
            print(f"      ✗  Value must be ≥ {lo}.")
            continue
        if hi is not None and val > hi:
            print(f"      ✗  Value must be ≤ {hi}.")
            continue
        return val


def _ask_bool(prompt_text: str, default: bool) -> bool:
    """Prompt user for a yes/no answer."""
    default_str = "yes" if default else "no"
    while True:
        raw = input(f"    {prompt_text} (yes/no) [{default_str}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes", "1", "true"):
            return True
        if raw in ("n", "no", "0", "false"):
            return False
        print("      ✗  Please type 'yes' or 'no' (or press Enter for default).")


def _ask_choice(prompt_text: str, options: list, default: str) -> str:
    """Prompt user to pick from a list of string options."""
    opts_str = " / ".join(options)
    while True:
        raw = input(f"    {prompt_text} ({opts_str}) [{default}]: ").strip().lower()
        if raw == "":
            return default
        if raw in options:
            return raw
        print(f"      ✗  Please choose one of: {opts_str}")


def _section(title: str):
    """Print a formatted section header."""
    width = 68
    print("\n" + "─" * width)
    print(f"  {title}")
    print("─" * width)
    print("  (Press Enter to accept the default value shown in [brackets])\n")


def _divider(label: str = ""):
    if label:
        print(f"\n  ── {label} " + "─" * max(0, 50 - len(label)))
    else:
        print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INTERACTIVE SESSION
# ─────────────────────────────────────────────────────────────────────────────

def get_user_inputs() -> dict:
    """
    Walk the user through all input groups interactively.
    Each prompt shows the literature/industrial default in brackets.
    Returns a dict with keys: reactor_params, feed, model_opts, run_opts.
    """

    print("\n" + "=" * 68)
    print("   METHANOL SYNTHESIS REACTOR — DIGITAL TWIN")
    print("   Interactive Input Session")
    print("=" * 68)
    print("\n  Defaults shown in [brackets] are industrially representative values")
    print("  for a Lurgi-type MRP reactor loaded with MegaMax Cu/ZnO/Al₂O₃.\n")
    print("  Literature sources for defaults:")
    print("    Reactor geometry : Lurgi MRP / Air Liquide reactor handbook")
    print("    Catalyst props   : Süd-Chemie / Clariant MegaMax datasheet")
    print("    Feed conditions  : Typical industrial syngas at 75 bar, 260 °C")
    print("    Kinetics         : Graaf et al. (1988/1990), Park et al. (2014)")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION A: REACTOR GEOMETRY
    # ─────────────────────────────────────────────────────────────────────────
    _section("SECTION A — REACTOR GEOMETRY  (Tube & Shell)")

    print("  The reactor is a multi-tubular fixed-bed type (BWR, Lurgi MRP style).")
    print("  Catalyst is packed on the TUBE side; boiling water on the SHELL side.\n")

    L = _ask_float(
        "Tube length                                         L  [m]",
        default=7.0, lo=1.0, hi=20.0)

    d_t = _ask_float(
        "Tube inner diameter                               d_t  [m]",
        default=0.038, lo=0.010, hi=0.100)

    N_tubes = _ask_int(
        "Number of tubes                               N_tubes  [-]",
        default=5000, lo=1, hi=50000)

    t_wall = _ask_float(
        "Tube wall thickness                            t_wall  [m]",
        default=0.003, lo=0.001, hi=0.010)

    k_wall = _ask_float(
        "Tube wall thermal conductivity (steel)         k_wall  [W/(m·K)]",
        default=50.0, lo=10.0, hi=120.0)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION B: CATALYST BED PROPERTIES
    # ─────────────────────────────────────────────────────────────────────────
    _section("SECTION B — CATALYST BED PROPERTIES  (MegaMax Cu/ZnO/Al₂O₃)")

    print("  Defaults correspond to Clariant MegaMax 800 pellets (6 mm equiv. sphere).")
    print("  Adjust rho_bulk if using a different catalyst lot or shape.\n")

    d_p = _ask_float(
        "Catalyst particle equivalent diameter            d_p  [m]",
        default=0.006, lo=0.001, hi=0.020)

    eps = _ask_float(
        "Bed void fraction (inter-particle)               eps  [-]",
        default=0.40, lo=0.30, hi=0.60)

    rho_bulk = _ask_float(
        "Catalyst bulk density                       rho_bulk  [kg/m³]",
        default=1100.0, lo=400.0, hi=2000.0)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION C: HEAT TRANSFER & SHELL-SIDE CONDITIONS
    # ─────────────────────────────────────────────────────────────────────────
    _section("SECTION C — HEAT TRANSFER & SHELL-SIDE CONDITIONS")

    print("  The shell-side coolant is boiling water (BWR = Boiling Water Reactor).")
    print("  T_shell sets the saturation temperature (= reactor operating temperature")
    print("  for an isothermal run).  Typical industrial range: 240–270 °C.\n")

    T_shell_C = _ask_float(
        "Shell-side (coolant) temperature              T_shell  [°C]",
        default=260.0, lo=200.0, hi=300.0)
    T_shell = T_shell_C + 273.15   # convert to Kelvin internally

    h_shell = _ask_float(
        "Shell-side boiling HTC                        h_shell  [W/(m²·K)]",
        default=8000.0, lo=1000.0, hi=25000.0)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION D: FEED CONDITIONS
    # ─────────────────────────────────────────────────────────────────────────
    _section("SECTION D — FEED CONDITIONS  (per tube, molar flow rates)")

    print("  All molar flows are per single tube [mol/s].")
    print("  The inlet temperature is typically equal to T_shell for near-isothermal")
    print("  pre-heating.  Pressure is total system pressure at reactor inlet.\n")

    T_in_C = _ask_float(
        "Inlet gas temperature                            T_in  [°C]",
        default=260.0, lo=150.0, hi=310.0)
    T_in = T_in_C + 273.15

    P_in_bar = _ask_float(
        "Inlet pressure                                   P_in  [bar]",
        default=75.0, lo=20.0, hi=150.0)
    P_in = P_in_bar * 1e5   # Pa internally

    _divider("Syngas composition")
    print("  Typical industrial syngas: H₂/(CO+CO₂) ≈ 2.0–3.1  (stoich. for MeOH = 2.0)")
    print("  CO₂/(CO+CO₂) ≈ 0.10–0.25  (higher CO₂ favours R2 over R1)\n")

    F_CO = _ask_float(
        "CO  molar flow (carbon monoxide)                F_CO  [mol/s]",
        default=0.240, lo=0.0, hi=10.0)

    F_CO2 = _ask_float(
        "CO₂ molar flow (carbon dioxide)                F_CO2  [mol/s]",
        default=0.055, lo=0.0, hi=5.0)

    F_H2 = _ask_float(
        "H₂  molar flow (hydrogen)                       F_H2  [mol/s]",
        default=0.600, lo=0.05, hi=20.0)

    F_N2 = _ask_float(
        "N₂  molar flow (inert diluent)                   F_N2  [mol/s]",
        default=0.040, lo=0.0, hi=2.0)

    _divider("Trace / recycle impurities")
    print("  Small amounts of H₂O and MeOH are present in industrial recycle loops.\n")

    F_H2O = _ask_float(
        "H₂O molar flow (trace water)                   F_H2O  [mol/s]",
        default=0.003, lo=0.0, hi=0.5)

    F_MeOH = _ask_float(
        "CH₃OH molar flow (recycle impurity)           F_MeOH  [mol/s]",
        default=0.001, lo=0.0, hi=0.5)

    # Live stoichiometry feedback
    ratio_H2   = F_H2  / max(F_CO + F_CO2, 1e-12)
    ratio_CO2  = F_CO2 / max(F_CO + F_CO2, 1e-12)
    stoich_ok  = 2.0 <= ratio_H2 <= 3.1
    print(f"\n  ── Feed stoichiometry check ──────────────────────────────────")
    print(f"    H₂ / (CO + CO₂)  = {ratio_H2:.3f}  "
          f"{'✓ in range [2.0 – 3.1]' if stoich_ok else '⚠ outside typical range [2.0–3.1]'}")
    print(f"    CO₂ / (CO + CO₂) = {ratio_CO2:.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION E: KINETICS MODEL & PHYSICAL SUB-MODELS
    # ─────────────────────────────────────────────────────────────────────────
    _section("SECTION E — KINETICS MODEL & PHYSICAL SUB-MODELS")

    print("  Available kinetics models for R1–R3 (CO/CO₂ hydrogenation + WGS):")
    print()
    print("    graaf  — Graaf et al., CES 43 (1988) / CES 45 (1990) [Netherlands]")
    print("             Dual CO+CO₂ pathway · LHHW form · Nestler (2020) MegaMax")
    print("             calibration · recommended for CO-rich syngas")
    print()
    print("    park   — Park et al., Fuel 118 (2014) [South Korea]")
    print("             Same Graaf LHHW form · re-fitted on commercial MegaMax")
    print("             lower Ea₂ (11.6 kJ/mol) → CO₂ path more T-sensitive")
    print("             recommended for CO₂-rich / modern syngas")
    print()
    print("    vbf    — Vanden Bussche & Froment, J. Catal. 161 (1996) [Belgium]")
    print("             CO₂-ONLY primary path · reverse-WGS coupling")
    print("             ⚠  K₃ inhibition strong at high H₂O — reduce F_H2O for")
    print("                fair comparison at pressures above ~50 bar")
    print()

    ACTIVE_MODEL = _ask_choice(
        "Select kinetics model                                  ",
        options=['graaf', 'park', 'vbf'], default='graaf')

    _divider("Physical sub-models")
    print()

    isothermal = _ask_bool(
        "Isothermal mode (True = boiling-water shell controls T)",
        default=True)

    use_pr_eos = _ask_bool(
        "Use Peng-Robinson EOS for fugacity coefficients        ",
        default=True)

    use_eta = _ask_bool(
        "Apply internal effectiveness factors (Thiele modulus)  ",
        default=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION F: SIMULATION & OUTPUT OPTIONS
    # ─────────────────────────────────────────────────────────────────────────
    _section("SECTION F — SIMULATION & OUTPUT OPTIONS")

    n_points = _ask_int(
        "ODE integration output points (resolution)  n_points  [-]",
        default=600, lo=50, hi=2000)

    run_sensitivity = _ask_bool(
        "Run one-at-a-time sensitivity analysis?               ",
        default=True)

    run_comparison = _ask_bool(
        "Run all-three-model kinetics comparison?              ",
        default=True)

    save_plot = _ask_bool(
        "Save reactor profile figure to PNG?                   ",
        default=True)

    plot_filename = "reactor_profiles.png"
    if save_plot:
        raw_fn = input("    Plot filename                       [reactor_profiles.png]: ").strip()
        if raw_fn:
            plot_filename = raw_fn if raw_fn.endswith('.png') else raw_fn + '.png'

    export_csv = _ask_bool(
        "Export axial profiles to CSV?                         ",
        default=True)

    csv_filename = "reactor_profiles.csv"
    if export_csv:
        raw_fn = input("    CSV filename                        [reactor_profiles.csv]: ").strip()
        if raw_fn:
            csv_filename = raw_fn if raw_fn.endswith('.csv') else raw_fn + '.csv'

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIRM & ECHO ALL INPUTS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  INPUT SUMMARY — confirm before running")
    print("=" * 68)

    print(f"\n  [A] Reactor Geometry")
    print(f"      Tube length                  L        = {L} m")
    print(f"      Tube inner diameter          d_t      = {d_t*1000:.1f} mm  ({d_t} m)")
    print(f"      Number of tubes              N_tubes  = {N_tubes:,}")
    print(f"      Wall thickness               t_wall   = {t_wall*1000:.1f} mm")
    print(f"      Wall conductivity            k_wall   = {k_wall} W/(m·K)")

    print(f"\n  [B] Catalyst Bed")
    print(f"      Particle diameter            d_p      = {d_p*1000:.1f} mm  ({d_p} m)")
    print(f"      Void fraction                eps      = {eps}")
    print(f"      Bulk density                 rho_bulk = {rho_bulk} kg/m³")

    print(f"\n  [C] Heat Transfer / Shell")
    print(f"      Shell temperature            T_shell  = {T_shell_C} °C  ({T_shell:.2f} K)")
    print(f"      Shell-side HTC               h_shell  = {h_shell:.0f} W/(m²·K)")

    print(f"\n  [D] Feed Conditions")
    print(f"      Inlet temperature            T_in     = {T_in_C} °C  ({T_in:.2f} K)")
    print(f"      Inlet pressure               P_in     = {P_in_bar} bar  ({P_in:.2e} Pa)")
    print(f"      F_CO                                  = {F_CO} mol/s")
    print(f"      F_CO2                                 = {F_CO2} mol/s")
    print(f"      F_H2                                  = {F_H2} mol/s")
    print(f"      F_N2                                  = {F_N2} mol/s")
    print(f"      F_H2O                                 = {F_H2O} mol/s")
    print(f"      F_MeOH                                = {F_MeOH} mol/s")
    print(f"      H₂/(CO+CO₂) ratio                    = {ratio_H2:.3f}")

    print(f"\n  [E] Kinetics & Sub-Models")
    print(f"      Kinetics model               model    = {ACTIVE_MODEL}")
    print(f"      Isothermal mode                       = {isothermal}")
    print(f"      Peng-Robinson EOS                     = {use_pr_eos}")
    print(f"      Effectiveness factors                 = {use_eta}")

    print(f"\n  [F] Simulation Options")
    print(f"      ODE output points            n_points = {n_points}")
    print(f"      Run sensitivity analysis              = {run_sensitivity}")
    print(f"      Run model comparison                  = {run_comparison}")
    print(f"      Save plot                             = {save_plot}  → {plot_filename}")
    print(f"      Export CSV                            = {export_csv}  → {csv_filename}")

    print()
    proceed = _ask_bool("Proceed with simulation?", default=True)
    if not proceed:
        print("\n  Simulation cancelled by user. No changes made.\n")
        raise SystemExit(0)

    return {
        'reactor_params': {
            'd_t':      d_t,
            'L':        L,
            'N_tubes':  N_tubes,
            't_wall':   t_wall,
            'k_wall':   k_wall,
            'd_p':      d_p,
            'eps':      eps,
            'rho_bulk': rho_bulk,
            'T_shell':  T_shell,
            'h_shell':  h_shell,
        },
        'feed': {
            'T_in':    T_in,
            'P_in':    P_in,
            'F_CO':    F_CO,
            'F_CO2':   F_CO2,
            'F_H2':    F_H2,
            'F_N2':    F_N2,
            'F_H2O':   F_H2O,
            'F_MeOH':  F_MeOH,
        },
        'model_opts': {
            'kinetics_model': ACTIVE_MODEL,
            'isothermal':     isothermal,
            'use_pr_eos':     use_pr_eos,
            'use_eta':        use_eta,
        },
        'run_opts': {
            'n_points':        n_points,
            'run_sensitivity': run_sensitivity,
            'run_comparison':  run_comparison,
            'save_plot':       save_plot,
            'plot_filename':   plot_filename,
            'export_csv':      export_csv,
            'csv_filename':    csv_filename,
        },
    }


# =============================================================================
# SECTION 11: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # ── Collect all inputs interactively ──────────────────────────────────────
    user = get_user_inputs()
    rp   = user['reactor_params']
    fd   = user['feed']
    mo   = user['model_opts']
    ro   = user['run_opts']

    # ── Initialise Digital Twin ───────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  INITIALISING DIGITAL TWIN")
    print("=" * 68)

    twin = DigitalTwin(
        reactor_params = rp,
        feed           = fd,
        kinetics_model = mo['kinetics_model'],
        isothermal     = mo['isothermal'],
        use_pr_eos     = mo['use_pr_eos'],
        use_eta        = mo['use_eta'],
    )

    print(f"\n  Kinetics model : {twin.kin.MODEL_NAME}")
    print(f"  Reference      : {twin.kin.MODEL_REF}")
    print(f"  Mode           : {'ISOTHERMAL — boiling water shell' if mo['isothermal'] else 'ADIABATIC / non-isothermal'}")
    print(f"  EOS            : {'Peng-Robinson fugacities' if mo['use_pr_eos'] else 'Ideal gas'}")
    print(f"  Effectiveness  : {'Thiele/Biot internal η factors' if mo['use_eta'] else 'η = 1 (no diffusion limit)'}")
    print(f"\n  Reaction network solved:")
    print("    R1  CO  + 2H₂   ↔  CH₃OH                  ΔH = −90.7  kJ/mol")
    print("    R2  CO₂ + 3H₂   ↔  CH₃OH + H₂O            ΔH = −49.5  kJ/mol")
    print("    R3  CO  + H₂O   ↔  CO₂  + H₂   [WGS]      ΔH = −41.2  kJ/mol")
    print("    R4  2CH₃OH       ↔  DME  + H₂O  [DME]      ΔH = −23.4  kJ/mol")
    print("    R5  CO  + 3H₂   →  CH₄  + H₂O  [methan.]  ΔH = −206.2 kJ/mol")
    print("    R6  2CO + 4H₂   ↔  EtOH + H₂O  [EtOH]     ΔH = −253.6 kJ/mol")
    print("    R7  3CO + 6H₂   ↔  PrOH + 2H₂O [PrOH]     ΔH = −417.3 kJ/mol")

    # ── Solve ─────────────────────────────────────────────────────────────────
    print(f"\n  Solving ODE (BDF, stiff) — {ro['n_points']} output points ...")
    profiles = twin.solve(n_points=ro['n_points'])
    print("  ODE integration complete.")

    # ── Outlet summary ────────────────────────────────────────────────────────
    twin.print_summary()

    # ── Kinetics model comparison ─────────────────────────────────────────────
    if ro['run_comparison']:
        print("\n  Running kinetics model comparison (Graaf / Park / VBF)...")
        if mo['kinetics_model'] == 'vbf':
            print("  Note: VBF at high H₂O/high-P — rates may be inhibited (see Section E).")
        compare_kinetics_models(rp, fd, models=['graaf', 'park', 'vbf'], n_points=300)

    # ── Sensitivity analysis ──────────────────────────────────────────────────
    if ro['run_sensitivity']:
        print("\n  Running one-at-a-time sensitivity analysis...")
        sensitivity_analysis(twin)

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\n  Generating reactor profile plots...")
    fig = plot_reactor_profiles(
        profiles,
        save_fig  = ro['save_plot'],
        filename  = ro['plot_filename'],
    )

    # ── CSV export ────────────────────────────────────────────────────────────
    if ro['export_csv']:
        import csv as _csv
        with open(ro['csv_filename'], 'w', newline='') as f:
            writer = _csv.writer(f)
            header = ['z_m', 'T_C', 'P_bar'] + \
                     [f'F_{sp}_mols' for sp in SPECIES] + \
                     ['X_CO', 'X_CO2', 'S_MeOH', 'S_DME', 'S_CH4', 'STY_MeOH_kgkgh']
            writer.writerow(header)
            for k in range(len(profiles['z'])):
                row = [profiles['z'][k],
                       profiles['T'][k] - 273.15,
                       profiles['P_bar'][k]]
                row += [profiles['F'][i, k] for i in range(N_SPECIES)]
                row += [profiles['X_CO'][k],  profiles['X_CO2'][k],
                        profiles['S_MeOH'][k], profiles['S_DME'][k],
                        profiles['S_CH4'][k],  profiles['STY_MeOH'][k]]
                writer.writerow(row)
        print(f"  Axial profiles exported → {ro['csv_filename']}")

    print("\n" + "=" * 68)
    print("  DIGITAL TWIN RUN COMPLETE")
    print("=" * 68)
    print("  Sub-models active:")
    print("    Thermodynamics  : Shomate Cp | Kirchhoff dHr | Graaf Keq")
    print("    Kinetics        : LHHW (" + twin.kin.MODEL_NAME + ")")
    print("    Equation of State: " + ("Peng-Robinson fugacities" if mo['use_pr_eos'] else "Ideal gas"))
    print("    Transport       : Chapman-Enskog | Wilke mixing | Zehner-Schlünder λ_eff")
    print("    Pressure drop   : Ergun equation")
    print("    Effectiveness   : " + ("Thiele modulus + Biot number" if mo['use_eta'] else "η = 1 (disabled)"))
    print("    Energy balance  : " + ("Isothermal (dT/dz = 0)" if mo['isothermal'] else "Full axial energy balance"))
    print("=" * 68 + "\n")
