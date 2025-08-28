import numpy as np
import jax
import jax.numpy as jnp
import equinox
from diffrax import (diffeqsolve, Kvaerno5, KenCarp5,
                     ODETerm, MultiTerm, SaveAt, PIDController, TqdmProgressMeter)
import diffrax as dfx
import optimistix as optx
from pymoo.core.problem import ElementwiseProblem

lb_1S = np.array([0,0,1])
ub_1S = np.array([3,3,50])
knames_1S = ['wA', 'wB', 'wE']

@equinox.filter_jit
def onestrainlinxpODE(T, Y, args):
# ======================================================================= #
#                                                                         #
# Single strain model for the following pathway:                          #
#      xS --[-- iS --> iA ----> iB --]--> xB                              #
#                                                                         #
# (c) 2025 Alexander PS Darlington, a.darlington.1@warwick.ac.uk          #
# ======================================================================= #
#
# ===== Current values ===================================================
#
    hPR, cPR = args
    Y = jnp.maximum(Y, 0)
# --- Population species --------------------------------------------------
    xS = Y[0];                                                                 # extracellular substrate
    xA = Y[1];                                                                 # extracellular product molecule A
    xB = Y[2];                                                                 # extracellular product molecule B
    N  = Y[3];                                                                 # population
    
    # --- Strain species ------------------------------------------------------
    iS = Y[4]; iE = Y[5];                                                      # internal substrate, internal energy
    mT = Y[6]; cT = Y[7]; pT = Y[8];                                           # transporter mRNA, translation complex, protein
    mE = Y[9]; cE = Y[10]; pE = Y[11];                                         # enzyme mRNA, translation complex, protein
    mH = Y[12]; cH = Y[13]; pH = Y[14];                                        # host q-protein mRNA, translation complex, protein
    mR = Y[15]; cR = Y[16]; pR = Y[17];                                        # r-protein mRNA, translation complex, protein
    rr = Y[18]; R  = Y[19];                                                    # rRNA, free ribosomes
    mA = Y[20]; cA = Y[21]; pA = Y[22]; iA = Y[23];                            # heterologous gene A mRNA, translation complex, protein, intracellular product A
    mB = Y[24]; cB = Y[25]; pB = Y[26]; iB = Y[27];                            # heterologous gene B mRNA, translation complex, protein, intracellular product B
    
    ## ===== Chassis parameters ================================================
    phie = hPR[0];                                                             # nutrient quality
    vIxS = hPR[1]; vEe = hPR[2]; kIxS = hPR[3]; kEe = hPR[4];                  # M-M parameters
    wT = hPR[5]; wE = hPR[6]; wH = hPR[7]; wR = hPR[8]; wr = hPR[9];           # transcription rates
    oX = hPR[10]; oR = hPR[11];                                                # transcription energy thresholds
    nX = hPR[12]; nR = hPR[13];                                                # protein lengths in amino acids
    bX = hPR[14]; uX = hPR[15];                                                # RBS dynamics
    brho = hPR[16]; urho = hPR[17];                                            # rRNA dynamics
    dymX = hPR[18];                                                            # mRNA decay rate
    maxG = hPR[19]; kappaG = hPR[20];                                          # peptide elongation rate parameters
    M0 = hPR[21];                                                              # cell size
    kH = hPR[22]; hH = hPR[23];                                                # q-protein feedback
    
    ## ==== Circuit parameters ================================================
    wA  = cPR[0]; nA  = cPR[1]; bA  = cPR[2];                                  # Protein A transcription rate, protein length, RBS strength
    vAe = cPR[3]; kAe = cPR[4]; vIxA = cPR[5]; vXiA = cPR[6];                  # Protein A kcat, kM, import rate, export rate
    wB  = cPR[7]; nB  = cPR[8]; bB  = cPR[9];                                 # Protein B transcription rate, protein length, RBS strength
    vBe = cPR[10]; kBe = cPR[11]; vXiB = cPR[12];                              # Protein B kcat, kM, import rate, export rate
    
    ## ===== Calculate rates ==================================================
    ### --- Global translation rates --------------------------------------------
    gammaX = (maxG*iE)/(kappaG + iE);
    
    ### --- Growth rates ---------------------------------------------------------
    lambda_ = (1/M0)*gammaX*(cT + cE + cH + cR + cA + cB);
    
    ### --- Transcription rates -------------------------------------------------
    g2mT = (wT*iE)/(oX + iE);
    g2mE = (wE*iE)/(oX + iE);
    g2mH = ((wH*iE)/(oX + iE))*(1/(1+(pH/kH)**hH));
    g2mR = (wR*iE)/(oR + iE);
    g2rr = (wr*iE)/(oR + iE);
    g2mA = (wA*iE)/(oX + iE);
    g2mB = (wB*iE)/(oX + iE);
    
    ### --- Translation rates ---------------------------------------------------
    m2pT = (gammaX/nX)*cT;
    m2pE = (gammaX/nX)*cE;
    m2pH = (gammaX/nX)*cH;
    m2pR = (gammaX/nR)*cR;
    m2pA = (gammaX/nA)*cA;
    m2pB = (gammaX/nB)*cB;
    
    ### ---- Metabolic rates ----------------------------------------------------
    ### xS --[-- iS --> iA --]--> xA --[--> iA --> iB --]--> xB
    xS2iS = (vIxS*xS*pT)/(kIxS + xS);
    iS2iE = (vEe*iS*pE)/(kEe + iS);
    iS2iA = (vAe*iS*pA)/(kAe + iS); # iS ---> iA
    iA2xA = vXiA*iA; # iA -]-> xA
    xA2iA = vIxA*xA; # xA -[-> iA
    iA2iB = (vBe*iA*pB)/(kBe + iA); # iA ---> iB
    iB2xB = vXiB*iB; # iB -]-> xB
    
    ## ===== Cellular model ===================================================
    ### --- Host metabolism ODEs ------------------------------------------------
    diS = xS2iS - iS2iE - iS2iA - lambda_*iS;
    deE = phie*iS2iE - lambda_*iE - nR*m2pR - nX*m2pT - nX*m2pE - nX*m2pH - nA*m2pA - nB*m2pB;
    
    ### --- Host proteome ODEs --------------------------------------------------
    dmT = g2mT - (lambda_ + dymX)*mT + m2pT - bX*R*mT + uX*cT;
    dcT = - lambda_*cT + bX*R*mT - uX*cT - m2pT;
    dpT = m2pT - lambda_*pT;
    dmE = g2mE - (lambda_ + dymX)*mE + m2pE - bX*R*mE + uX*cE;
    dcE = - lambda_*cE + bX*R*mE - uX*cE - m2pE;
    dpE = m2pE - lambda_*pE;
    dmH = g2mH - (lambda_ + dymX)*mH + m2pH - bX*R*mH + uX*cH;
    dcH = - lambda_*cH + bX*R*mH - uX*cH - m2pH;
    dpH = m2pH - lambda_*pH;
    dmR = g2mR - (lambda_ + dymX)*mR + m2pR - bX*R*mR + uX*cR;
    dcR = - lambda_*cR + bX*R*mR - uX*cR - m2pR;
    dpR = m2pR - lambda_*pR - brho*pR*rr + urho*R;
    drr = g2rr - lambda_*rr - brho*pR*rr + urho*R;
    dR  = brho*pR*rr - urho*R - lambda_*R \
          + m2pT - bX*R*mT + uX*cT \
          + m2pE - bX*R*mE + uX*cE \
          + m2pH - bX*R*mH + uX*cH \
          + m2pR - bX*R*mR + uX*cR \
          + m2pA - bA*R*mA + uX*cA \
          + m2pB - bB*R*mB + uX*cB;
    
    ## ===== Pathway ODEs =====================================================
    ### --- Circuit gene ODEs ---------------------------------------------------
    dmA = g2mA - (lambda_ + dymX)*mA + m2pA - bA*R*mA + uX*cA;
    dcA = - lambda_*cA + bA*R*mA - uX*cA - m2pA;
    dpA = m2pA - lambda_*pA;
    dmB = g2mB - (lambda_ + dymX)*mB + m2pB - bB*R*mB + uX*cB;
    dcB = - lambda_*cB + bB*R*mB - uX*cB - m2pB;
    dpB = m2pB - lambda_*pB;
    
    ### --- Circuit species ODEs ------------------------------------------------
    diA = iS2iA + xA2iA - iA2xA - iA2iB - lambda_*iA;
    diB = iA2iB - iB2xB - lambda_*iB;
    
    ## ===== Culture ODEs =====================================================
    dxS = - xS2iS*N;
    dxA = iA2xA*N - xA2iA*N;
    dxB = iB2xB*N;
    dN  = lambda_*N;
    
    ## ===== Return ===========================================================
    dY_by_dt = jnp.array([dxS,dxA,dxB,dN,diS,deE,
        dmT,dcT,dpT,dmE,dcE,dpE,dmH,dcH,dpH,
        dmR,dcR,dpR,drr,dR,
        dmA,dcA,dpA,diA,dmB,dcB,dpB,diB]);

    population    = jnp.array([N]);
    growth_rate   = jnp.array([lambda_]);
    extracellular_metabolism = jnp.array([xS,xA,xB]);

    dY_by_dt = jnp.where(Y<=0, jnp.maximum(dY_by_dt, 0), dY_by_dt)

    return (dY_by_dt, population, growth_rate, extracellular_metabolism)

@equinox.filter_jit
def onestrainlinxpODE_wrapped(T, Y, args):
    dy_dt, _, _, _ = onestrainlinxpODE(T, Y, args)
    return dy_dt

W_onestrainlinxpODE = lambda t, y, arg1, arg2: onestrainlinxpODE_wrapped(t, y, (arg1, arg2))

@equinox.filter_jit
def explicit_term(t, y, args):
    return 0.0

# Optimisation
## One strain, multi-objective
solver = KenCarp5(root_finder=dfx.VeryChord(rtol=1e-3, atol=1e-6, norm=optx.rms_norm))
term = MultiTerm(ODETerm(explicit_term), ODETerm(onestrainlinxpODE_wrapped))
stepsize_controller = PIDController(rtol=1e-3, atol=1e-6, pcoeff=0.2, icoeff=0.5)
max_steps = 8192

#@equinox.filter_jit
def oneStrainMultiObj(hPR, cPR, Y00, N0, tmax, scalefactor):
    # Set initial conditions
    Y0 = Y00; Y0[3] = N0
    Y0 = jnp.maximum(Y0, 0)

    def true_func(sol):
        ys = jnp.asarray(sol.ys)
        T = sol.ts
        Y = jnp.maximum(ys, 0)

        xS = Y[:,0]; 
        xB = Y[:,2]; 
        pA = Y[:,22]; 
        pB = Y[:,26]; 

        # Get outputs
        # --- Population species --------------------------------------------------
        #xS = Y[:,0];                                                                 # extracellular substrate
        #xA = Y[:,1];                                                                 # extracellular product molecule A
        #xB = Y[:,2];                                                                 # extracellular product molecule B
        #N  = Y[:,3];                                                                 # population
        
        ## --- Strain species ------------------------------------------------------
        #iS = Y[:,4]; iE = Y[:,5];                                                      # internal substrate, internal energy
        #mT = Y[:,6]; cT = Y[:,7]; pT = Y[:,8];                                           # transporter mRNA, translation complex, protein
        #mE = Y[:,9]; cE = Y[:,10]; pE = Y[:,11];                                         # enzyme mRNA, translation complex, protein
        #mH = Y[:,12]; cH = Y[:,13]; pH = Y[:,14];                                        # host q-protein mRNA, translation complex, protein
        #mR = Y[:,15]; cR = Y[:,16]; pR = Y[:,17];                                        # r-protein mRNA, translation complex, protein
        #rr = Y[:,18]; R  = Y[:,19];                                          # rRNA, free ribosomes
        #mA = Y[:,20]; cA = Y[:,21]; pA = Y[:,22]; iA = Y[:,23];                            # heterologous gene A mRNA, translation complex, protein, intracellular product A
        #mB = Y[:,24]; cB = Y[:,25]; pB = Y[:,26]; iB = Y[:,27];                            # heterologous gene B mRNA, translation complex, protein, intracellular product B
    
        # Performance metrics
        idx = jnp.sum(xS > 1e-6) - 1
        p_yield = xB[-1]/xS[0]
        p_prod = xB[idx]/T[idx]
        output = jnp.array([p_yield, p_prod])
        nA = cPR[1]; nB = cPR[8]; M0 = hPR[21]
        phiAB = jnp.max((1/M0) * (nA * pA + nB * pB))

        #if output[0] > 1:
        #    cost = np.array([0.,0.])
        #else:
        #    cost = - output/scalefactor
        cond = output[0] > 1
        cost = jax.lax.cond(cond, 
                            lambda x: jnp.array([0.,0.]),
                            lambda x: -x/scalefactor,
                            output)
        return cost, output, T, Y, hPR, cPR, phiAB

    def false_func(sol):
        return jnp.array([jnp.inf, jnp.inf]), jnp.array([jnp.inf, jnp.inf]), jnp.zeros_like(sol.ts), jnp.zeros_like(sol.ys), \
               jnp.zeros_like(hPR), jnp.zeros_like(cPR), jnp.max(0.)

    # Solve ODE
    tsave = jnp.linspace(0, tmax, max_steps)
    saveat = SaveAt(ts=tsave)

    sol = diffeqsolve(term, solver=solver, y0=Y0, t0=0, t1=tmax, args=(hPR, cPR), dt0=None,
                    stepsize_controller=stepsize_controller, max_steps=max_steps, saveat=saveat, throw=False)

    cond = sol.result == dfx.RESULTS.successful
    return jax.lax.cond(cond, true_func, false_func, sol)

def W_oneStrainMultiObj(x, args):
    hPR, cPR, Y10, N0, tmax, scalefactor = args
    cPR[0] = 10**x[0] # Set wA
    cPR[7] = 10**x[1] # Set wB
    hPR[6] = x[2] # Set wE
    cost, output, _, _, _, _, _  = oneStrainMultiObj(hPR, cPR, Y10, N0, tmax, scalefactor)
    return cost, output

## One strain, single objective
def oneStrainSingleObj(x, args):
    hPR, cPR, Y10, N0, tmax, scalefactor = args
    _, output = W_oneStrainMultiObj(x, args=(hPR, cPR, Y10, N0, tmax, jnp.array([1, 1])))
    cost = - jnp.sum(output * scalefactor)
    return cost, output

## Define the problem class for one strain multiple objectives
class oneStrainMultiObj_problem(ElementwiseProblem):
    def __init__(self, nvars, nobjs, lb, ub, **kwargs):
        super().__init__(n_var=nvars, n_obj=nobjs, n_constr=0,
                         xl=lb, xu=ub)
        for key, value in kwargs.items():
            self.__dict__[key] = value

    def _evaluate(self, x, out, *args, **kwargs):
        cost, _ = W_oneStrainMultiObj(x, self.args)
        out['F'] = jax.device_get(cost)

## Define the problem class for one strain single objective
class oneStrainSingleObj_problem(ElementwiseProblem):
    def __init__(self, nvars, nobjs, lb, ub, **kwargs):
        super().__init__(n_var=nvars, n_obj=nobjs, n_constr=0,
                         xl=lb, xu=ub)
        for key, value in kwargs.items():
            self.__dict__[key] = value

    def _evaluate(self, x, out, *args, **kwargs):
        cost, _ = oneStrainSingleObj(x, self.args)
        out['F'] = jax.device_get(cost)

def init_arrays(params, mparams, CellY0=None):
    hPR = np.array([params.phie0,
                    params.vIxS,params.vEe,
                    params.kIxS,params.kEe,
                    params.wT,params.wE,params.wH,params.wR,params.wr,
                    params.oX,params.oR,
                    params.nX,params.nR,
                    params.bX,params.uX,params.brho,params.urho,
                    params.dymX,
                    params.maxG,params.kappaG,params.M0,
                    params.kH,params.hH])
    
    cPR = np.array([params.wA,params.nA,params.bA,
                    params.vAe,params.kAe,
                    params.vIxA,params.vXiA,
                    params.wB,params.nB,params.bB,
                    params.vBe,params.kBe,params.vXiB])

    if CellY0 is None:
        # Initital conditions for simulating steady state
        Y0 = np.array([params.initxS0,0,0,0,
                       params.iS0,params.eE0,0,0,
                       params.pT0,0,0,
                       params.pE0,0,0,
                       params.pH0,0,0,
                       params.pR0,params.rr0,params.R0,0,0,
                       params.pA0,0,0,0,
                       params.pB0,0])
        cPR[0], cPR[7] = 0, 0
    else:
        Y0 = np.hstack((np.array([params.xS0,params.xA0,params.xB0,params.N0]),
                        CellY0,
                        np.array([0,0,params.pA0,0,0,0,params.pB0,0])))
    
    return Y0, hPR, cPR

def update_arrays(params, Y0):
    Y0 = np.maximum(Y0, 0)
    Y0[0:4] = params.xS0, params.xA0, params.xB0, params.N0

    return Y0

ONE_STRAIN_UTILS = {
    "name": "one strain (1S)",
    "short_name": "1S",
    "init_arrays": init_arrays,
    "update_arrays": update_arrays,
    "mparams": None,
    "ode": onestrainlinxpODE,
    "ode_scipy": W_onestrainlinxpODE,
    "ode_diffrax": onestrainlinxpODE_wrapped,
    "calc_objs": W_oneStrainMultiObj,
    "sop": oneStrainSingleObj_problem,
    "mop": oneStrainMultiObj_problem,
    "var_bounds": [lb_1S, ub_1S],
}
