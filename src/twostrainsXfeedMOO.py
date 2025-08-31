import numpy as np
import jax
import jax.numpy as jnp
import equinox
from diffrax import (diffeqsolve, Kvaerno5, KenCarp5,
                     ODETerm, MultiTerm, SaveAt, PIDController, TqdmProgressMeter)
import diffrax as dfx
import optimistix as optx
from pymoo.core.problem import ElementwiseProblem
from system_parameters import TwostrainsXfeedParameters

lb_2S = np.array([0,0,0,1,1])
ub_2S = np.array([3,3,1,50,50])
knames_2S = ['wA', 'wB', 'r0', 'wE_a', 'wE_b']

@equinox.filter_jit
def twostrainsXfeedLinxpODE(T, Y, args):
# ======================================================================= #
#                                                                         #
# Two strain model for the following pathway with crossfeeding:           #
#    xS --[-- iS --> iA --]--> xA                                         #
#                              xA --[-- iA --> iB --]--> xB               #
#                                                                         #
# (c) 2025 Alexander PS Darlington, a.darlington.1@warwick.ac.uk          #
# ======================================================================= #
    
    ## ===== Current values ===================================================
    hPR, cPR = args
    Y = jnp.maximum(Y, 0)

    ## --- Population species --------------------------------------------------
    xS  = Y[0];                                                                # extracellular substrate
    xA  = Y[1];                                                                # extracellular product molecule A
    xB  = Y[2];                                                                # extracellular product molecule B
    N_a = Y[3];                                                                # population of strain A
    N_b = Y[4];                                                                # population of strain B
   
    ### --- Strain A ------------------------------------------------------------
    iS_a = Y[5]; iEA_a  = Y[6];                                                # internal substrate, internal energy
    mT_a = Y[7]; cT_a  = Y[8]; pT_a  = Y[9];                                   # transporter mRNA, translation complex, protein
    mE_a = Y[10]; cE_a = Y[11]; pE_a = Y[12];                                  # enzyme mRNA, translation complex, protein
    mH_a = Y[13]; cH_a = Y[14]; pH_a = Y[15];                                  # host q-protein mRNA, translation complex, protein
    mR_a = Y[16]; cR_a = Y[17]; pR_a = Y[18];                                  # r-protein mRNA, translation complex, protein
    rr_a = Y[19]; R_a  = Y[20];                                                # rRNA, free ribosomes 
    mA_a = Y[21]; cA_a = Y[22]; pA_a = Y[23]; iA_a = Y[24];                    # heterologous gene A mRNA, translation complex, protein, intracellular product A
    
    ### --- Strain B ------------------------------------------------------------
    iS_b = Y[25]; iEB_b = Y[26];                                               # internal substrate, internal energy
    mT_b = Y[27]; cT_b = Y[28]; pT_b = Y[29];                                  # transporter mRNA, translation complex, protein
    mE_b = Y[30]; cE_b = Y[31]; pE_b = Y[32];                                  # enzyme mRNA, translation complex, protein
    mH_b = Y[33]; cH_b = Y[34]; pH_b = Y[35];                                  # host q-protein mRNA, translation complex, protein
    mR_b = Y[36]; cR_b = Y[37]; pR_b = Y[38];                                  # r-protein mRNA, translation complex, protein
    rr_b = Y[39]; R_b  = Y[40];                                                # rRNA, free ribosomes 
    mB_b = Y[41]; cB_b = Y[42]; pB_b = Y[43]; iA_b = Y[44]; iB_b = Y[45];      # heterologous gene B mRNA, translation complex, protein, intracellular product B

    ## --- Extracellular x-feed metabolites ------------------------------------
    xEA   = Y[46];                                                             # extracellular energy produced by strain A
    xEB   = Y[47];                                                             # extracellular energy produced by strain B
    iEB_a = Y[48];                                                             # intracellular energy molecule B inside strain A
    iEA_b = Y[49];   
    
    # ===== Chassis parameters ================================================
    ## --- Strain A ------------------------------------------------------------
    phie_a = hPR[0];                                                           # nutrient quality
    vIxS_a = hPR[1]; vEe_a   = hPR[2]; kIxS_a = hPR[3]; kEe_a = hPR[4];        # M-M parameters
    wT_a   = hPR[5]; wE_a    = hPR[6]; wH_a   = hPR[7]; wR_a  = hPR[8]; wr_a = hPR[9]; # transcription rates
    oX_a   = hPR[10]; oR_a   = hPR[11];                                        # transcription energy thresholds
    nX_a   = hPR[12]; nR_a   = hPR[13];                                        # protein lengths in amino acids
    bX_a   = hPR[14]; uX_a   = hPR[15];                                        # RBS dynamics
    brho_a = hPR[16]; urho_a = hPR[17];                                        # rRNA dynamics
    dymX_a = hPR[18];                                                          # mRNA decay rate
    maxG_a = hPR[19]; kG_a = hPR[20];                                          # peptide elongation rate parameters
    M0_a   = hPR[21];                                                          # cell size
    kH_a   = hPR[22]; hH_a   = hPR[23];                                        # q-protein feedback
    
    ## --- Strain B ------------------------------------------------------------
    phie_b = hPR[24];                                                          # nutrient quality
    vIxS_b = hPR[25]; vEe_b  = hPR[26]; kIxS_b = hPR[27]; kEe_b = hPR[28];     # M-M parameters
    wT_b   = hPR[29]; wE_b   = hPR[30]; wH_b   = hPR[31]; wR_b  = hPR[32]; wr_b = hPR[33]; # transcription rates
    oX_b   = hPR[34]; oR_b   = hPR[35];                                        # transcription energy thresholds
    nX_b   = hPR[36]; nR_b   = hPR[37];                                        # protein lengths in amino acids
    bX_b   = hPR[38]; uX_b   = hPR[39];                                        # RBS dynamics
    brho_b = hPR[40]; urho_b = hPR[41];                                        # rRNA dynamics
    dymX_b = hPR[42];                                                          # mRNA decay rate
    maxG_b = hPR[43]; kG_b   = hPR[44];                                        # peptide elongation rate parameters
    M0_b   = hPR[45];                                                          # cell size
    kH_b   = hPR[46]; hH_b   = hPR[47];                                        # q-protein feedback

    #--- Cross feeding parameters --------------------------------------------
    vXiEA_a = hPR[48];                                                         # iEA export from strain A
    vIiEB_a = hPR[49];                                                         # iEB import to strain A
    vIxEA_b = hPR[50];                                                         # iEA import to strain B
    vXiEB_b = hPR[51];

    # ==== Circuit parameters ================================================
    wA  = cPR[0]; nA   = cPR[1]; bA    = cPR[2];                               # Protein A transcription rate, protein length, RBS strength
    vAe = cPR[3]; kAe  = cPR[4]; vIxA  = cPR[5]; vXiA = cPR[6];                # Protein A kcat, kM, import rate, export rate
    wB  = cPR[7]; nB   = cPR[8]; bB    = cPR[9];                               # Protein B transcription rate, protein length, RBS strength
    vBe = cPR[10]; kBe = cPR[11]; vXiB = cPR[12];                            # Protein B kcat, kM, import rate, export rate
    
    # ===== Calculate Strain A rates =========================================
    ## --- Global translation rates --------------------------------------------
    gammaX_a = (maxG_a*iEB_a)/(kG_a + iEB_a);
    
    ## --- Growth rates ---------------------------------------------------------
    lambda_a = (1/M0_a)*gammaX_a*(cT_a + cE_a + cH_a + cR_a + cA_a);
    
    ## --- Transcription rates -------------------------------------------------
    g2mT_a = (wT_a*iEB_a)/(oX_a + iEB_a);
    g2mE_a = (wE_a*iEB_a)/(oX_a + iEB_a);
    g2mH_a = ((wH_a*iEB_a)/(oX_a + iEB_a))*(1/(1+(pH_a/kH_a)**hH_a));
    g2mR_a = (wR_a*iEB_a)/(oR_a + iEB_a);
    g2rr_a = (wr_a*iEB_a)/(oR_a + iEB_a);
    g2mA_a = (wA*iEB_a)/(oX_a + iEB_a);
    
    ## --- Translation rates ---------------------------------------------------
    m2pT_a = (gammaX_a/nX_a)*cT_a;
    m2pE_a = (gammaX_a/nX_a)*cE_a;
    m2pH_a = (gammaX_a/nX_a)*cH_a;
    m2pR_a = (gammaX_a/nR_a)*cR_a;
    m2pA_a = (gammaX_a/nA)*cA_a;
    
    ## ---- Metabolic rates ----------------------------------------------------
    xS2iS_a   = (vIxS_a*xS*pT_a)/(kIxS_a + xS);
    iS2iEA_a  = (vEe_a*iS_a*pE_a)/(kEe_a + iS_a);
    iEA2xEA_a = (vXiEA_a*iEA_a);
    xEB2iEB_a = (vIiEB_a*xEB);
    
    ## ---- Pathway rates ------------------------------------------------------
    # xS --[-- iS --> iA --]--> xA
    iS2iA_a = (vAe*iS_a*pA_a)/(kAe + iS_a); # iS ---> iA
    iA2xA_a = (vXiA*iA_a);                  # iA -]-> xA
    
    # ===== Calculate Strain B rates =========================================
    ## --- Global translation rates --------------------------------------------
    gammaX_b = (maxG_b*iEA_b)/(kG_b + iEA_b);
    
    ## --- Growth rates ---------------------------------------------------------
    lambda_b = (1/M0_b)*gammaX_b*(cT_b + cE_b + cH_b + cR_b + cB_b);
    
    ## --- Transcription rates -------------------------------------------------
    g2mT_b = (wT_b*iEA_b)/(oX_b + iEA_b);
    g2mE_b = (wE_b*iEA_b)/(oX_b + iEA_b);
    g2mH_b = ((wH_b*iEA_b)/(oX_b + iEA_b))*(1/(1+(pH_b/kH_b)**hH_b));
    g2mR_b = (wR_b*iEA_b)/(oR_b + iEA_b);
    g2rr_b = (wr_b*iEA_b)/(oR_b + iEA_b);
    g2mB_b = (wB*iEA_b)/(oX_b + iEA_b);
    
    ## --- Translation rates ---------------------------------------------------
    m2pT_b = (gammaX_b/nX_b)*cT_b;
    m2pE_b = (gammaX_b/nX_b)*cE_b;
    m2pH_b = (gammaX_b/nX_b)*cH_b;
    m2pR_b = (gammaX_b/nR_b)*cR_b;
    m2pB_b = (gammaX_b/nB)*cB_b;
    
    ## ---- Metabolic rates ----------------------------------------------------
    xS2iS_b   = (vIxS_b*xS*pT_b)/(kIxS_b + xS);
    iS2iEB_b  = (vEe_b*iS_b*pE_b)/(kEe_b + iS_b);
    xEA2iEA_b = (vIxEA_b*xEA);
    iEB2xEB_b = (vXiEB_b*iEB_b);
    
    ## --- Pathway rates -------------------------------------------------------
    # xA --[--> iA --> iB --]--> xB
    xA2iA_b = (vIxA*xA);                    # xA -[-> iA
    iA2iB_b = (vBe*iA_b*pB_b)/(kBe + iA_b); # iA ---> iB
    iB2xB_b = (vXiB*iB_b);                  # iB -]-> xB
    
    # ===== Strain A cellular model ==========================================
    ## --- Host metabolism ODEs ------------------------------------------------
    diS_a = xS2iS_a - iS2iEA_a - iS2iA_a - lambda_a*iS_a;
    diEA_a = phie_a*iS2iEA_a - lambda_a*iEA_a - iEA2xEA_a;
    diEB_a = xEB2iEB_a - lambda_a*iEB_a \
         - nR_a*m2pR_a - nX_a*m2pT_a - nX_a*m2pE_a - nX_a*m2pH_a - nA*m2pA_a;
    
    ## --- Host proteome ODEs --------------------------------------------------
    dmT_a = g2mT_a - (lambda_a + dymX_a)*mT_a + m2pT_a - bX_a*R_a*mT_a + uX_a*cT_a;
    dcT_a = - lambda_a*cT_a + bX_a*R_a*mT_a - uX_a*cT_a - m2pT_a;
    dpT_a = m2pT_a - lambda_a*pT_a;
    dmE_a = g2mE_a - (lambda_a + dymX_a)*mE_a + m2pE_a - bX_a*R_a*mE_a + uX_a*cE_a;
    dcE_a = - lambda_a*cE_a + bX_a*R_a*mE_a - uX_a*cE_a - m2pE_a;
    dpE_a = m2pE_a - lambda_a*pE_a;
    dmH_a = g2mH_a - (lambda_a + dymX_a)*mH_a + m2pH_a - bX_a*R_a*mH_a + uX_a*cH_a;
    dcH_a = - lambda_a*cH_a + bX_a*R_a*mH_a - uX_a*cH_a - m2pH_a;
    dpH_a = m2pH_a - lambda_a*pH_a;
    dmR_a = g2mR_a - (lambda_a + dymX_a)*mR_a + m2pR_a - bX_a*R_a*mR_a + uX_a*cR_a;
    dcR_a = - lambda_a*cR_a + bX_a*R_a*mR_a - uX_a*cR_a - m2pR_a;
    dpR_a = m2pR_a - lambda_a*pR_a - brho_a*pR_a*rr_a + urho_a*R_a;
    drr_a = g2rr_a - lambda_a*rr_a - brho_a*pR_a*rr_a + urho_a*R_a;
    dR_a  = brho_a* pR_a*rr_a - urho_a*R_a - lambda_a*R_a \
        + m2pT_a - bX_a*R_a*mT_a + uX_a*cT_a \
        + m2pE_a - bX_a*R_a*mE_a + uX_a*cE_a \
        + m2pH_a - bX_a*R_a*mH_a + uX_a*cH_a \
        + m2pR_a - bX_a*R_a*mR_a + uX_a*cR_a \
        + m2pA_a - bA*R_a*mA_a + uX_a*cA_a;
    
    # ===== Strain B cellular model ==========================================
    ## --- Host metabolism ODEs ------------------------------------------------
    diS_b  = xS2iS_b - iS2iEB_b - lambda_b*iS_b;
    diEA_b = xEA2iEA_b - lambda_b*iEA_b \
        - nR_b*m2pR_b - nX_b*m2pT_b - nX_b*m2pE_b - nX_b*m2pH_b - nB*m2pB_b;
    diEB_b = phie_b*iS2iEB_b - lambda_b*iEB_b - iEB2xEB_b;
    
    ## --- Host proteome ODEs --------------------------------------------------
    dmT_b = g2mT_b - (lambda_b + dymX_b)*mT_b + m2pT_b - bX_b*R_b*mT_b + uX_b*cT_b;
    dcT_b = - lambda_b*cT_b + bX_b*R_b*mT_b - uX_b*cT_b - m2pT_b;
    dpT_b = m2pT_b - lambda_b*pT_b;
    dmE_b = g2mE_b - (lambda_b + dymX_b)*mE_b + m2pE_b - bX_b*R_b*mE_b + uX_b*cE_b;
    dcE_b = - lambda_b*cE_b + bX_b*R_b*mE_b - uX_b*cE_b - m2pE_b;
    dpE_b = m2pE_b - lambda_b*pE_b;
    dmH_b = g2mH_b - (lambda_b + dymX_b)*mH_b + m2pH_b - bX_b*R_b*mH_b + uX_b*cH_b;
    dcH_b = - lambda_b*cH_b + bX_b*R_b*mH_b - uX_b*cH_b - m2pH_b;
    dpH_b = m2pH_b - lambda_b*pH_b;
    dmR_b = g2mR_b - (lambda_b + dymX_b)*mR_b + m2pR_b - bX_b*R_b*mR_b + uX_b*cR_b;
    dcR_b = - lambda_b*cR_b + bX_b*R_b*mR_b - uX_b*cR_b - m2pR_b;
    dpR_b = m2pR_b - lambda_b*pR_b - brho_b*pR_b*rr_b + urho_b*R_b;
    drr_b = g2rr_b - lambda_b*rr_b - brho_b*pR_b*rr_b + urho_b*R_b;
    dR_b  = brho_b*pR_b*rr_b - urho_b*R_b - lambda_b*R_b \
        + m2pT_b - bX_b*R_b*mT_b + uX_b*cT_b \
        + m2pE_b - bX_b*R_b*mE_b + uX_b*cE_b \
        + m2pH_b - bX_b*R_b*mH_b + uX_b*cH_b \
        + m2pR_b - bX_b*R_b*mR_b + uX_b*cR_b \
        + m2pB_b - bB*R_b*mB_b + uX_b*cB_b;
    
    # ===== Pathway ODEs =====================================================
    ## --- Strain A circuit and pathway ODEs -----------------------------------
    dmA_a = g2mA_a - (lambda_a + dymX_a)*mA_a + m2pA_a - bA*R_a*mA_a + uX_a*cA_a;
    dcA_a = - lambda_a*cA_a + bA*R_a*mA_a - uX_a*cA_a - m2pA_a;
    dpA_a = m2pA_a - lambda_a*pA_a;
    diA_a = iS2iA_a - iA2xA_a - lambda_a*iA_a;
    
    ## --- Strain B Circuit and pathway ODEs -----------------------------------
    dmB_b = g2mB_b - (lambda_b + dymX_b)*mB_b + m2pB_b - bB*R_b*mB_b + uX_b*cB_b;
    dcB_b = - lambda_b*cB_b + bB*R_b*mB_b - uX_b*cB_b - m2pB_b;
    dpB_b = m2pB_b - lambda_b*pB_b;
    diA_b = xA2iA_b - iA2iB_b - lambda_b*iA_b;
    diB_b = iA2iB_b - iB2xB_b - lambda_b*iB_b;
    
    # ===== Culture ODEs =====================================================
    dxS  = - xS2iS_a*N_a - xS2iS_b*N_b;
    dxA  = iA2xA_a*N_a - xA2iA_b*N_b;
    dxB  = iB2xB_b*N_b;
    dN_a = lambda_a*N_a;
    dN_b = lambda_b*N_b;
    dxEA = + iEA2xEA_a*N_a - xEA2iEA_b*N_b;
    dxEB = - xEB2iEB_a*N_a + iEB2xEB_b*N_b;
    
    # ===== Return ===========================================================
    dY_by_dt = jnp.array([dxS,dxA,dxB,dN_a,dN_b,diS_a,diEA_a,dmT_a,dcT_a,
                          dpT_a,dmE_a,dcE_a,dpE_a,dmH_a,dcH_a,dpH_a,dmR_a,
		                  dcR_a,dpR_a,drr_a,dR_a,dmA_a,dcA_a,dpA_a,diA_a,
                          diS_b,diEB_b,dmT_b,dcT_b,dpT_b,dmE_b,dcE_b,dpE_b,
		                  dmH_b,dcH_b,dpH_b,dmR_b,dcR_b,dpR_b,drr_b,dR_b,
		                  dmB_b,dcB_b,dpB_b,diA_b,diB_b,dxEA,dxEB,
		                  diEB_a,diEA_b])
    
    population    = jnp.array([N_a, N_b]);
    growth_rate   = jnp.array([lambda_a, lambda_b]);
    extracellular_metabolism = jnp.array([xS, xA, xB]);
	
    dY_by_dt = jnp.where(Y<=0, jnp.maximum(dY_by_dt, 0), dY_by_dt)

    return (dY_by_dt, population, growth_rate, extracellular_metabolism)

def twostrainsXfeedLinxpODE_wrapped(T, Y, args):
    dy_dt, _, _, _ = twostrainsXfeedLinxpODE(T, Y, args)
    return dy_dt

W_twostrainsXfeedLinxpODE = lambda T, Y, arg1, arg2: twostrainsXfeedLinxpODE_wrapped(T, Y, (arg1, arg2))

@equinox.filter_jit
def explicit_term(t, y, args):
    return 0.0

## Two strains with cross feeding, multi-objective
solver = KenCarp5(root_finder=dfx.VeryChord(rtol=1e-3, atol=1e-6, norm=optx.rms_norm))
term = MultiTerm(ODETerm(explicit_term), ODETerm(twostrainsXfeedLinxpODE_wrapped))
stepsize_controller = PIDController(rtol=1e-3, atol=1e-6, pcoeff=0.3, icoeff=0.4)
max_steps = 8192

@equinox.filter_jit
def twoStrainXfeedMultiObj(hPR, cPR, Y00, r0, N0, tmax, scalefactor):
    # Initial conditions
    Y0 = Y00; Y0.at[3].set(r0*N0); Y0.at[4].set((1 - r0)*N0);
    Y0 = jnp.maximum(Y0, 0);

    def true_func(sol):
        ys = jnp.array(sol.ys)
        T = sol.ts
        Y = jnp.maximum(ys, 0)
    
        xS  = Y[:,0]; 
        xB  = Y[:,2]; 
        pA_a = Y[:,23];
        pB_b = Y[:,43];

        # Get outputs
        ### --- Population species --------------------------------------------------
        #xS  = Y[:,0];                                                                # extracellular substrate
        #xA  = Y[:,1];                                                                # extracellular product molecule A
        #xB  = Y[:,2];                                                                # extracellular product molecule B
        #N_a = Y[:,3];                                                                # population of strain A
        #N_b = Y[:,4];                                                                # population of strain B
        
        #### --- Strain A ------------------------------------------------------------
        #iS_a = Y[:,5]; iE_a  = Y[:,6];                                                 # internal substrate, internal energy
        #mT_a = Y[:,7]; cT_a  = Y[:,8]; pT_a  = Y[:,9];                                   # transporter mRNA, translation complex, protein
        #mE_a = Y[:,10]; cE_a = Y[:,11]; pE_a = Y[:,12];                                  # enzyme mRNA, translation complex, protein
        #mH_a = Y[:,13]; cH_a = Y[:,14]; pH_a = Y[:,15];                                  # host q-protein mRNA, translation complex, protein
        #mR_a = Y[:,16]; cR_a = Y[:,17]; pR_a = Y[:,18];                                  # r-protein mRNA, translation complex, protein
        #rr_a = Y[:,19]; R_a  = Y[:,20];                                                # rRNA, free ribosomes 
        #mA_a = Y[:,21]; cA_a = Y[:,22]; pA_a = Y[:,23]; iA_a = Y[:,24];                    # heterologous gene A mRNA, translation complex, protein, intracellular product A
        
        #### --- Strain B ------------------------------------------------------------
        #iS_b = Y[:,25]; iE_b = Y[:,26];                                                # internal substrate, internal energy
        #mT_b = Y[:,27]; cT_b = Y[:,28]; pT_b = Y[:,29];                                  # transporter mRNA, translation complex, protein
        #mE_b = Y[:,30]; cE_b = Y[:,31]; pE_b = Y[:,32];                                  # enzyme mRNA, translation complex, protein
        #mH_b = Y[:,33]; cH_b = Y[:,34]; pH_b = Y[:,35];                                  # host q-protein mRNA, translation complex, protein
        #mR_b = Y[:,36]; cR_b = Y[:,37]; pR_b = Y[:,38];                                  # r-protein mRNA, translation complex, protein
        #rr_b = Y[:,39]; R_b  = Y[:,40];                                                # rRNA, free ribosomes 
        #mB_b = Y[:,41]; cB_b = Y[:,42]; pB_b = Y[:,43]; iA_b = Y[:,44]; iB_b = Y[:,45];      # heterologous gene B mRNA, translation complex, protein, intracellular product B
    
        # Performance metrics
        idx = jnp.sum(xS > 1e-6) - 1
        p_yield = xB[-1]/xS[0]
        p_prod = xB[idx]/T[idx]
        output = jnp.array([p_yield, p_prod])
    
        # Calculate mass fractions
        M0_a = hPR[21]; M0_b = hPR[45]
        nA = cPR[1]; nB = cPR[8]
        phiA = (1/M0_a) * (nA * pA_a)
        phiB = (1/M0_b) * (nB * pB_b)
        phiAB = jnp.array([phiA, phiB])
    
        #if output[0] > 1:
        #    cost = np.array([0,0])
        #else:
        #    cost = -output/scalefactor
        cond = output[0] > 1
        cost = jax.lax.cond(cond, 
                            lambda x: jnp.array([0., 0.]),
                            lambda x: -x/scalefactor,
                            output)
        return cost, output, T, Y, hPR, cPR, phiAB

    def false_func(sol):
        return jnp.array([jnp.inf, jnp.inf]), jnp.array([jnp.inf, jnp.inf]), jnp.zeros_like(sol.ts), jnp.zeros_like(sol.ys), \
               jnp.zeros_like(hPR), jnp.zeros_like(cPR), jnp.zeros((2, max_steps))

    # Solve ODE
    tsave = jnp.linspace(0, tmax, max_steps)
    saveat = SaveAt(ts=tsave)
    sol = diffeqsolve(term, solver=solver, y0=Y0, t0=0, t1=tmax, args=(hPR, cPR), dt0=None,
                    stepsize_controller=stepsize_controller, max_steps=max_steps, saveat=saveat, throw=False)

    cond = sol.result == dfx.RESULTS.successful
    return jax.lax.cond(cond, true_func, false_func, sol)

def W_twoStrainXfeedMultiObj(x, args):
        hPR, cPR, Y20, N0, tmax, scalefactor = args
        cPR[0] = 10**x[0] # Set wA
        cPR[7] = 10**x[1] # Set wB
        r0 = x[2]
        hPR[6] = x[3] # set wE_a
        hPR[30] = x[4] # set wE_b
        cost, output, _, _, _, _, _ = twoStrainXfeedMultiObj(hPR, cPR, Y20, r0, N0, tmax, scalefactor)
        return cost, output

## Two strains, single objective
def twoStrainXfeedSingleObj(x, args):
    # Simulate model
    hPR, cPR, Y20, N0, tmax, scalefactor = args
    _, output = W_twoStrainXfeedMultiObj(x, args=(hPR, cPR, Y20, N0, tmax, jnp.array([1, 1])))
    cost = - jnp.sum(output * scalefactor)

    return cost, output

## Define the problem class for two strains multiple objectives
class twoStrainXfeedMultiObj_problem(ElementwiseProblem):
    def __init__(self, nvars, nobjs, lb, ub, **kwargs):
        super().__init__(n_var=nvars, n_obj=nobjs, n_constr=0,
                         xl=lb, xu=ub)
        for key, value in kwargs.items():
            self.__dict__[key] = value

    def _evaluate(self, x, out, *args, **kwargs):
        cost, _ = W_twoStrainXfeedMultiObj(x, self.args)
        out['F'] = jax.device_get(cost)

## Define the problem class for two strain single objectives
class twoStrainXfeedSingleObj_problem(ElementwiseProblem):
    def __init__(self, nvars, nobjs, lb, ub, **kwargs):
        super().__init__(n_var=nvars, n_obj=nobjs, n_constr=0,
                         xl=lb, xu=ub)
        for key, value in kwargs.items():
            self.__dict__[key] = value

    def _evaluate(self, x, out, *args, **kwargs):
        cost, _ = twoStrainXfeedSingleObj(x, self.args)
        out['F'] = jax.device_get(cost)

def init_arrays(params, mparams, CellY0):
    Y0 = np.hstack((np.array([params.xS0,params.xA0,params.xB0,params.r0*params.N0, (1-params.r0)*params.N0]),
                    CellY0,
                    np.array([0,0,params.pA0,0]),
                    CellY0,
                    np.array([0,0,params.pB0,0,0,0,0,0,0])))

    hPR = np.array([params.phie0,params.vIxS,params.vEe,
                    params.kIxS,params.kEe,
                    params.wT,params.wE,params.wH,params.wR,params.wr,
                    params.oX,params.oR,
                    params.nX,params.nR,
                    params.bX,params.uX,params.brho,params.urho,params.dymX,
                    params.maxG,params.kappaG,params.M0,
                    params.kH,params.hH,
                    params.phie0,params.vIxS,params.vEe,
                    params.kIxS,params.kEe,
                    params.wT,params.wE,params.wH,params.wR,params.wr,
                    params.oX,params.oR,
                    params.nX,params.nR,
                    params.bX,params.uX,params.brho,params.urho,params.dymX,
                    params.maxG,params.kappaG,params.M0,
                    params.kH,params.hH,
                    mparams.vXiEA,mparams.vIiEB,mparams.vIxEA,mparams.vXiEB])
    
    cPR = np.array([params.wA,params.nA,params.bA,
                    params.vAe,params.kAe,params.vIxA,params.vXiA,
                    params.wB,params.nB,params.bB,
                    params.vBe,params.kBe,params.vXiB])
    
    return Y0, hPR, cPR

def update_arrays(params, Y0):
    Y0 = np.maximum(Y0, 0)
    Y0[0:5] = params.xS0, params.xA0, params.xB0, params.r0*params.N0, (1-params.r0)*params.N0

    return Y0

TWO_STRAINS_XFEED_UTILS = {
    "name": "2S with Xfeed",
    "short_name": "2XF",
    "init_arrays": init_arrays,
    "update_arrays": update_arrays,
    "mparams": TwostrainsXfeedParameters,
    "ode": twostrainsXfeedLinxpODE,
    "ode_scipy": W_twostrainsXfeedLinxpODE,
    "ode_diffrax": twostrainsXfeedLinxpODE_wrapped,
    "calc_objs": W_twoStrainXfeedMultiObj,
    "sop": twoStrainXfeedSingleObj_problem,
    "mop": twoStrainXfeedMultiObj_problem,
    "var_bounds": [lb_2S, ub_2S],
}
