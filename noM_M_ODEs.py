import equinox
import jax.numpy as jnp

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
    extracellular_metabolism = jnp.array([xS,xB]);

    dY_by_dt = jnp.where(Y<=0, jnp.maximum(dY_by_dt, 0), dY_by_dt)

    return (dY_by_dt, population, growth_rate, extracellular_metabolism)

@equinox.filter_jit
def onestrainlinxpODE_wrapped(T, Y, args):
    dy_dt, _, _, _ = onestrainlinxpODE(T, Y, args)
    return dy_dt

W_onestrainlinxpODE = lambda t, y, arg1, arg2: onestrainlinxpODE_wrapped(t, y, (arg1, arg2))

# ===== SIMULATE TWO STRAIN MODEL ========================================
@equinox.filter_jit
def twostrainslinxpODE(T, Y, args):
# ======================================================================= #
#                                                                         #
# Two strain model for the following pathway:                             #
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
    iS_a = Y[5]; iE_a  = Y[6];                                                 # internal substrate, internal energy
    mT_a = Y[7]; cT_a  = Y[8]; pT_a  = Y[9];                                   # transporter mRNA, translation complex, protein
    mE_a = Y[10]; cE_a = Y[11]; pE_a = Y[12];                                  # enzyme mRNA, translation complex, protein
    mH_a = Y[13]; cH_a = Y[14]; pH_a = Y[15];                                  # host q-protein mRNA, translation complex, protein
    mR_a = Y[16]; cR_a = Y[17]; pR_a = Y[18];                                  # r-protein mRNA, translation complex, protein
    rr_a = Y[19]; R_a  = Y[20];                                                # rRNA, free ribosomes 
    mA_a = Y[21]; cA_a = Y[22]; pA_a = Y[23]; iA_a = Y[24];                    # heterologous gene A mRNA, translation complex, protein, intracellular product A
    
    ### --- Strain B ------------------------------------------------------------
    iS_b = Y[25]; iE_b = Y[26];                                                # internal substrate, internal energy
    mT_b = Y[27]; cT_b = Y[28]; pT_b = Y[29];                                  # transporter mRNA, translation complex, protein
    mE_b = Y[30]; cE_b = Y[31]; pE_b = Y[32];                                  # enzyme mRNA, translation complex, protein
    mH_b = Y[33]; cH_b = Y[34]; pH_b = Y[35];                                  # host q-protein mRNA, translation complex, protein
    mR_b = Y[36]; cR_b = Y[37]; pR_b = Y[38];                                  # r-protein mRNA, translation complex, protein
    rr_b = Y[39]; R_b  = Y[40];                                                # rRNA, free ribosomes 
    mB_b = Y[41]; cB_b = Y[42]; pB_b = Y[43]; iA_b = Y[44]; iB_b = Y[45];      # heterologous gene B mRNA, translation complex, protein, intracellular product B
    
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
    
    # ==== Circuit parameters ================================================
    wA  = cPR[0]; nA   = cPR[1]; bA    = cPR[2];                               # Protein A transcription rate, protein length, RBS strength
    vAe = cPR[3]; kAe  = cPR[4]; vIxA  = cPR[5]; vXiA = cPR[6];                # Protein A kcat, kM, import rate, export rate
    wB  = cPR[7]; nB   = cPR[8]; bB    = cPR[9];                               # Protein B transcription rate, protein length, RBS strength
    vBe = cPR[10]; kBe = cPR[11]; vXiB = cPR[12];                            # Protein B kcat, kM, import rate, export rate
    
    # ===== Calculate Strain A rates =========================================
    ## --- Global translation rates --------------------------------------------
    gammaX_a = (maxG_a*iE_a)/(kG_a + iE_a);
    
    ## --- Growth rates ---------------------------------------------------------
    lambda_a = (1/M0_a)*gammaX_a*(cT_a + cE_a + cH_a + cR_a + cA_a);
    
    ## --- Transcription rates -------------------------------------------------
    g2mT_a = (wT_a*iE_a)/(oX_a + iE_a);
    g2mE_a = (wE_a*iE_a)/(oX_a + iE_a);
    g2mH_a = ((wH_a*iE_a)/(oX_a + iE_a))*(1/(1+(pH_a/kH_a)**hH_a));
    g2mR_a = (wR_a*iE_a)/(oR_a + iE_a);
    g2rr_a = (wr_a*iE_a)/(oR_a + iE_a);
    g2mA_a = (wA*iE_a)/(oX_a + iE_a);
    
    ## --- Translation rates ---------------------------------------------------
    m2pT_a = (gammaX_a/nX_a)*cT_a;
    m2pE_a = (gammaX_a/nX_a)*cE_a;
    m2pH_a = (gammaX_a/nX_a)*cH_a;
    m2xR_a = (gammaX_a/nR_a)*cR_a;
    m2pA_a = (gammaX_a/nA)*cA_a;
    
    ## ---- Metabolic rates ----------------------------------------------------
    xS2iS_a = (vIxS_a*xS*pT_a)/(kIxS_a + xS);
    iS2iE_a = (vEe_a*iS_a*pE_a)/(kEe_a + iS_a);
    
    # ---- Pathway rates ------------------------------------------------------
    # xS --[-- iS --> iA --]--> xA
    iS2iA_a = (vAe*iS_a*pA_a)/(kAe + iS_a); # iS ---> iA
    iA2xA_a = vXiA*iA_a; # iA -]-> xA
    
    # ===== Calculate Strain B rates =========================================
    ## --- Global translation rates --------------------------------------------
    gammaX_b = (maxG_b*iE_b)/(kG_b + iE_b);
    
    ## --- Growth rates ---------------------------------------------------------
    lambda_b = (1/M0_b)*gammaX_b*(cT_b + cE_b + cH_b + cR_b + cB_b);
    
    ## --- Transcription rates -------------------------------------------------
    g2mT_b = (wT_b*iE_b)/(oX_b + iE_b);
    g2mE_b = (wE_b*iE_b)/(oX_b + iE_b);
    g2mH_b = ((wH_b*iE_b)/(oX_b + iE_b))*(1/(1+(pH_b/kH_b)**hH_b));
    g2mR_b = (wR_b*iE_b)/(oR_b + iE_b);
    g2rr_b = (wr_b*iE_b)/(oR_b + iE_b);
    g2mB_b = (wB*iE_b)/(oX_b + iE_b);
    
    ## --- Translation rates ---------------------------------------------------
    m2pT_b = (gammaX_b/nX_b)*cT_b;
    m2pE_b = (gammaX_b/nX_b)*cE_b;
    m2pH_b = (gammaX_b/nX_b)*cH_b;
    m2xR_b = (gammaX_b/nR_b)*cR_b;
    m2pB_b = (gammaX_b/nB)*cB_b;
    
    ## ---- Metabolic rates ----------------------------------------------------
    xS2iS_b = (vIxS_b*xS*pT_b)/(kIxS_b + xS);
    iS2iE_b = (vEe_b*iS_b*pE_b)/(kEe_b + iS_b);
    
    ## --- Pathway rates -------------------------------------------------------
    # xA --[--> iA --> iB --]--> xB
    xA2iA_b = vIxA*xA; # xA -[-> iA
    iA2iB_b = (vBe*iA_b*pB_b)/(kBe + iA_b); # iA ---> iB
    iB2xB_b = vXiB*iB_b; # iB -]-> xB
    
    # ===== Strain A cellular model ==========================================
    ## --- Host metabolism ODEs ------------------------------------------------
    diS_a = xS2iS_a - iS2iE_a - iS2iA_a - lambda_a*iS_a;
    diE_a = phie_a*iS2iE_a - lambda_a*iE_a - nR_a*m2xR_a - nX_a*m2pT_a - nX_a*m2pE_a - nX_a*m2pH_a - nA*m2pA_a;
    
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
    dmR_a = g2mR_a - (lambda_a + dymX_a)*mR_a + m2xR_a - bX_a*R_a*mR_a + uX_a*cR_a;
    dcR_a = - lambda_a*cR_a + bX_a*R_a*mR_a - uX_a*cR_a - m2xR_a;
    dpR_a = m2xR_a - lambda_a*pR_a - brho_a*pR_a*rr_a + urho_a*R_a;
    drr_a = g2rr_a - lambda_a*rr_a - brho_a*pR_a*rr_a + urho_a*R_a;
    dR_a = brho_a*pR_a*rr_a - urho_a*R_a - lambda_a*R_a \
        + m2pT_a - bX_a*R_a*mT_a + uX_a*cT_a \
        + m2pE_a - bX_a*R_a*mE_a + uX_a*cE_a \
        + m2pH_a - bX_a*R_a*mH_a + uX_a*cH_a \
        + m2xR_a - bX_a*R_a*mR_a + uX_a*cR_a \
        + m2pA_a - bA*R_a*mA_a + uX_a*cA_a;
    
    # ===== Strain B cellular model ==========================================
    ## --- Host metabolism ODEs ------------------------------------------------
    diS_b = xS2iS_b - iS2iE_b - lambda_b*iS_b;
    diE_b = phie_b*iS2iE_b - lambda_b*iE_b - nR_b*m2xR_b - nX_b*m2pT_b - nX_b*m2pE_b - nX_b*m2pH_b - nB*m2pB_b;
    
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
    dmR_b = g2mR_b - (lambda_b + dymX_b)*mR_b + m2xR_b - bX_b*R_b*mR_b + uX_b*cR_b;
    dcR_b = - lambda_b*cR_b + bX_b*R_b*mR_b - uX_b*cR_b - m2xR_b;
    dpR_b = m2xR_b - lambda_b*pR_b - brho_b*pR_b*rr_b + urho_b*R_b;
    drr_b = g2rr_b - lambda_b*rr_b - brho_b*pR_b*rr_b + urho_b*R_b;
    dR_b = brho_b*pR_b*rr_b - urho_b*R_b - lambda_b*R_b \
        + m2pT_b - bX_b*R_b*mT_b + uX_b*cT_b \
        + m2pE_b - bX_b*R_b*mE_b + uX_b*cE_b \
        + m2pH_b - bX_b*R_b*mH_b + uX_b*cH_b \
        + m2xR_b - bX_b*R_b*mR_b + uX_b*cR_b \
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
    
    ## --- Culture ODEs --------------------------------------------------------
    dxS = - xS2iS_a*N_a - xS2iS_b*N_b;
    dxA = iA2xA_a*N_a - xA2iA_b*N_b;
    dxB = iB2xB_b*N_b;
    dN_a = lambda_a*N_a;
    dN_b = lambda_b*N_b;
    
    # ===== Return ===========================================================
    dY_by_dt = jnp.array([dxS,dxA,dxB,dN_a,dN_b,diS_a,diE_a,dmT_a,dcT_a,dpT_a,
           dmE_a,dcE_a,dpE_a,dmH_a,dcH_a,dpH_a,dmR_a,dcR_a,
		   dpR_a,drr_a,dR_a,dmA_a,dcA_a,dpA_a,diA_a,diS_b,diE_b,
           dmT_b,dcT_b,dpT_b,dmE_b,dcE_b,dpE_b,dmH_b,dcH_b,dpH_b,
           dmR_b,dcR_b,dpR_b,drr_b,dR_b,dmB_b,dcB_b,dpB_b,
		   diA_b,diB_b])

    dY_by_dt = jnp.where(Y<=0, jnp.maximum(dY_by_dt, 0), dY_by_dt)
    population    = jnp.array([N_a,N_b]);
    growth_rate   = jnp.array([lambda_a,lambda_b]);
    extracellular_metabolism = jnp.array([xS, xB]);
    intracellular_metabolism = jnp.array([iA_a, iA_b, 0, iB_b]);
    pathway_gene = jnp.array([mA_a, cA_a, pA_a, mB_b, cB_b, pB_b]);
    
    return (dY_by_dt, population, growth_rate, extracellular_metabolism, 
            intracellular_metabolism, pathway_gene)

@equinox.filter_jit
def twostrainslinxpODE_wrapped(T, Y, args):
    dy_dt, _, _, _, _, _ = twostrainslinxpODE(T, Y, args)
    return dy_dt

W_twostrainslinxpODE = lambda T, Y, arg1, arg2: twostrainslinxpODE_wrapped(T, Y, (arg1, arg2))

# ===== SIMULATE TWO STRAIN MODEL WITH CROSS FEEDING =====================
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
    extracellular_metabolism = jnp.array([xS, xB]);
	
    dY_by_dt = jnp.where(Y<=0, jnp.maximum(dY_by_dt, 0), dY_by_dt)

    return (dY_by_dt, population, growth_rate, extracellular_metabolism)

def twostrainsXfeedLinxpODE_wrapped(T, Y, args):
    dy_dt, _, _, _ = twostrainsXfeedLinxpODE(T, Y, args)
    return dy_dt

W_twostrainsXfeedLinxpODE = lambda T, Y, arg1, arg2: twostrainsXfeedLinxpODE_wrapped(T, Y, (arg1, arg2))

# ===== SIMULATE TWO STRAIN MODEL WITH QS-MEDIATED DECAY ========================================
@equinox.filter_jit
def twostrainsQSDinhlinxpODE(T, Y, args):
# ======================================================================= #
#  --- Reference -----------------------------------------------------------
#  Dinh, Chen and Prather 2022
#  https://dx.doi.org/10.1021/acssynbio.9b00451?ref=pdf
#
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
    iS_a = Y[5]; iE_a  = Y[6];                                                 # internal substrate, internal energy
    mT_a = Y[7]; cT_a  = Y[8]; pT_a  = Y[9];                                   # transporter mRNA, translation complex, protein
    mE_a = Y[10]; cE_a = Y[11]; pE_a = Y[12];                                  # enzyme mRNA, translation complex, protein
    mH_a = Y[13]; cH_a = Y[14]; pH_a = Y[15];                                  # host q-protein mRNA, translation complex, protein
    mR_a = Y[16]; cR_a = Y[17]; pR_a = Y[18];                                  # r-protein mRNA, translation complex, protein
    rr_a = Y[19]; R_a  = Y[20];                                                # rRNA, free ribosomes 
    mA_a = Y[21]; cA_a = Y[22]; pA_a = Y[23]; iA_a = Y[24];                    # heterologous gene A mRNA, translation complex, protein, intracellular product A
    
    ### --- Strain B ------------------------------------------------------------
    iS_b = Y[25]; iE_b = Y[26];                                                # internal substrate, internal energy
    mT_b = Y[27]; cT_b = Y[28]; pT_b = Y[29];                                  # transporter mRNA, translation complex, protein
    mE_b = Y[30]; cE_b = Y[31]; pE_b = Y[32];                                  # enzyme mRNA, translation complex, protein
    mH_b = Y[33]; cH_b = Y[34]; pH_b = Y[35];                                  # host q-protein mRNA, translation complex, protein
    mR_b = Y[36]; cR_b = Y[37]; pR_b = Y[38];                                  # r-protein mRNA, translation complex, protein
    rr_b = Y[39]; R_b  = Y[40];                                                # rRNA, free ribosomes 
    mB_b = Y[41]; cB_b = Y[42]; pB_b = Y[43]; iA_b = Y[44]; iB_b = Y[45];      # heterologous gene B mRNA, translation complex, protein, intracellular product B
    
    ### --- Strain A QS system --------------------------------------------------
    mI_a = Y[46]; cI_a = Y[47]; pI_a = Y[48];                                  # LuxI mRNA, translation complex, protein
    mU_a = Y[49]; cU_a = Y[50]; pU_a = Y[51]; pUa_a = Y[52];                   # LuxR mRNA, translation complex, inactive protein, active (qs-bound) protein 
    mP_a = Y[53]; cP_a = Y[54]; pP_a = Y[55];                                  # pE_a-killing protease mRNA, translation complex, protein
    iQA_a = Y[56]; iQB_a = Y[57];                                              # QS molecules inside strain A

    ### --- Strain B QS system --------------------------------------------------
    mI_b = Y[58]; cI_b = Y[59]; pI_b = Y[60];                                  # LuxI mRNA, translation complex, protein
    mU_b = Y[61]; cU_b = Y[62]; pU_b = Y[63]; pUa_b = Y[64];                   # LuxR mRNA, translation complex, inactive protein, active [qs-bound] protein 
    mP_b = Y[65]; cP_b = Y[66]; pP_b = Y[67];                                  # pE_b-killing protease mRNA, translation complex, protein
    iQA_b = Y[68]; iQB_b = Y[69];                                              # QS molecules inside strain B

    ### --- Extracellular QS molecules ------------------------------------------
    xQA = Y[70]; xQB = Y[71];                                                  # QS molecules

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

    ### --- New parameters ------------------------------------------------------
    wI_a    = hPR[48]; wU_a     = hPR[49]; wP_a = hPR[50];                     # transcription rates luxI, luxR and sRNA
    nI_a    = hPR[51]; nU_a     = hPR[52]; nP_a = hPR[53];                     # protein lengths in amino acids
    bI_a    = hPR[54]; bU_a     = hPR[55]; bP_a = hPR[56];                     # RBS binding rate
    bQU_a   = hPR[57]; uQU_a    = hPR[58];                                     # QS-LuxR binding rates
    kU_a    = hPR[59]; hU_a     = hPR[60];                                     # LuxR hill parameters
    phiQA_a = hPR[61]; dphiQA_a = hPR[62];                                     # QS
    vEiQA_a = hPR[63]; kEiQA_a  = hPR[64];                                     #
    vXiQA_a = hPR[65]; vIiQB_a  = hPR[66];                                     # iQA export rate, iQB import rate
    vP_a    = hPR[67]; kP_a     = hPR[68];                                     # sRNA decay rate, sRNA:mRNA binding rate
	
    wI_b    = hPR[69]; wU_b     = hPR[70]; wP_b = hPR[71];                     # transcription rates luxI, luxR and sRNA
    nI_b    = hPR[72]; nU_b     = hPR[73]; nP_b = hPR[74];                     # protein lengths in amino acids
    bI_b    = hPR[75]; bU_b     = hPR[76]; bP_b = hPR[77];                     # RBS binding rate
    bQU_b   = hPR[78]; uQU_b    = hPR[79];                                     # QS-LuxR binding rates
    kU_b    = hPR[80]; hU_b     = hPR[81];                                     # LuxR hill parameters
    phiQB_b = hPR[82]; dphiQB_b = hPR[83];                                     # QS
    vEiQB_b = hPR[84]; kEiQB_b  = hPR[85];                                     #
    vXiQB_b = hPR[86]; vIiQA_b  = hPR[87];                                     # iQA export rate, iQB import rate
    vP_b    = hPR[88]; kP_b     = hPR[89];                                     # sRNA decay rate, sRNA:mRNA binding rate

    # ==== Circuit parameters ================================================
    wA  = cPR[0]; nA   = cPR[1]; bA    = cPR[2];                               # Protein A transcription rate, protein length, RBS strength
    vAe = cPR[3]; kAe  = cPR[4]; vIxA  = cPR[5]; vXiA = cPR[6];                # Protein A kcat, kM, import rate, export rate
    wB  = cPR[7]; nB   = cPR[8]; bB    = cPR[9];                               # Protein B transcription rate, protein length, RBS strength
    vBe = cPR[10]; kBe = cPR[11]; vXiB = cPR[12];                              # Protein B kcat, kM, import rate, export rate
    
    # ===== Calculate Strain A rates =========================================
    ## --- Global translation rates --------------------------------------------
    gammaX_a = (maxG_a*iE_a)/(kG_a + iE_a);
    
    ## --- Growth rates ---------------------------------------------------------
    lambda_a = (1/M0_a)*gammaX_a*(cT_a + cE_a + cH_a + cR_a + cA_a + cI_a + cU_a + cP_a);
    
    ## --- Transcription rates -------------------------------------------------
    g2mT_a = (wT_a*iE_a)/(oX_a + iE_a);
    g2mE_a = (wE_a*iE_a)/(oX_a + iE_a);
    g2mH_a = ((wH_a*iE_a)/(oX_a + iE_a))*(1/(1+(pH_a/kH_a)**hH_a));
    g2mR_a = (wR_a*iE_a)/(oR_a + iE_a);
    g2rr_a = (wr_a*iE_a)/(oR_a + iE_a);
    g2mA_a = (wA*iE_a)/(oX_a + iE_a);
    g2mI_a = (wI_a*iE_a)/(oX_a + iE_a);
    g2mU_a = (wU_a*iE_a)/(oX_a + iE_a);
    g2mP_a = ((wP_a*iE_a)/(oX_a + iE_a))*((pUa_a**hU_a)/((kU_a**hU_a) + (pUa_a**hU_a)));

    ## --- Translation rates ---------------------------------------------------
    m2pT_a = (gammaX_a/nX_a)*cT_a;
    m2pE_a = (gammaX_a/nX_a)*cE_a;
    m2pH_a = (gammaX_a/nX_a)*cH_a;
    m2xR_a = (gammaX_a/nR_a)*cR_a;
    m2pA_a = (gammaX_a/nA)*cA_a;
    m2pI_a = (gammaX_a/nI_a)*cI_a;
    m2pU_a = (gammaX_a/nU_a)*cU_a;
    m2pP_a = (gammaX_a/nP_a)*cP_a;

    ## ---- Metabolic rates ----------------------------------------------------
    xS2iS_a = (vIxS_a*xS*pT_a)/(kIxS_a + xS);
    iS2iE_a = (vEe_a*iS_a*pE_a)/(kEe_a + iS_a);
    iS2iQA_a  = (vEiQA_a*iS_a*pI_a)/(kEiQA_a + iS_a);
    iQA2xQA_a = (vXiQA_a*iQA_a);
    xQB2iQB_a = (vIiQB_a*xQB);
    pE2oo_a = (vP_a*pE_a*pP_a)/(kP_a + pE_a);
    # ---- Pathway rates ------------------------------------------------------
    # xS --[-- iS --> iA --]--> xA
    iS2iA_a = (vAe*iS_a*pA_a)/(kAe + iS_a); # iS ---> iA
    iA2xA_a = vXiA*iA_a; # iA -]-> xA
    
    # ===== Calculate Strain B rates =========================================
    ## --- Global translation rates --------------------------------------------
    gammaX_b = (maxG_b*iE_b)/(kG_b + iE_b);
    
    ## --- Growth rates ---------------------------------------------------------
    lambda_b = (1/M0_b)*gammaX_b*(cT_b + cE_b + cH_b + cR_b + cB_b + cI_b + cU_b + cP_b);
    
    ## --- Transcription rates -------------------------------------------------
    g2mT_b = (wT_b*iE_b)/(oX_b + iE_b);
    g2mE_b = (wE_b*iE_b)/(oX_b + iE_b);
    g2mH_b = ((wH_b*iE_b)/(oX_b + iE_b))*(1/(1+(pH_b/kH_b)**hH_b));
    g2mR_b = (wR_b*iE_b)/(oR_b + iE_b);
    g2rr_b = (wr_b*iE_b)/(oR_b + iE_b);
    g2mB_b = (wB*iE_b)/(oX_b + iE_b);
    g2mI_b = (wI_b*iE_b)/(oX_b + iE_b);
    g2mU_b = (wU_b*iE_b)/(oX_b + iE_b);
    g2mP_b = ((wP_b*iE_b)/(oX_b + iE_b))*(((pUa_b/kU_b)**hU_b)/(1 + ((pUa_b/kU_b)**hU_b)));
    
    ## --- Translation rates ---------------------------------------------------
    m2pT_b = (gammaX_b/nX_b)*cT_b;
    m2pE_b = (gammaX_b/nX_b)*cE_b;
    m2pH_b = (gammaX_b/nX_b)*cH_b;
    m2xR_b = (gammaX_b/nR_b)*cR_b;
    m2pB_b = (gammaX_b/nB)*cB_b;
    m2pI_b = (gammaX_b/nI_b)*cI_b;
    m2pU_b = (gammaX_b/nU_b)*cU_b;
    m2pP_b = (gammaX_b/nP_b)*cP_b;

    ## ---- Metabolic rates ----------------------------------------------------
    xS2iS_b = (vIxS_b*xS*pT_b)/(kIxS_b + xS);
    iS2iE_b = (vEe_b*iS_b*pE_b)/(kEe_b + iS_b);
    iS2iQB_b  = (vEiQB_b*iS_b*pI_b)/(kEiQB_b + iS_b);
    iQB2xQB_b = (vXiQB_b*iQB_b);
    xQA2iQA_b = (vIiQA_b*xQA);
    pE2oo_b = (vP_b*pE_b*pP_b)/(kP_b + pE_b);

    ## --- Pathway rates -------------------------------------------------------
    # xA --[--> iA --> iB --]--> xB
    xA2iA_b = vIxA*xA; # xA -[-> iA
    iA2iB_b = (vBe*iA_b*pB_b)/(kBe + iA_b); # iA ---> iB
    iB2xB_b = vXiB*iB_b; # iB -]-> xB
    
    # ===== Strain A cellular model ==========================================
    ## --- Host metabolism ODEs ------------------------------------------------
    diS_a = xS2iS_a - iS2iE_a - iS2iA_a - lambda_a*iS_a - dphiQA_a*iS2iQA_a;
    diE_a = phie_a*iS2iE_a - lambda_a*iE_a - nR_a*m2xR_a - nX_a*m2pT_a - nX_a*m2pE_a - nX_a*m2pH_a - nA*m2pA_a \
        - nI_a*m2pI_a - nU_a*m2pU_a - nP_a*m2pP_a;
    
    ## --- Host proteome ODEs --------------------------------------------------
    dmT_a = g2mT_a - (lambda_a + dymX_a)*mT_a + m2pT_a - bX_a*R_a*mT_a + uX_a*cT_a;
    dcT_a = - lambda_a*cT_a + bX_a*R_a*mT_a - uX_a*cT_a - m2pT_a;
    dpT_a = m2pT_a - lambda_a*pT_a;
    dmE_a = g2mE_a - (lambda_a + dymX_a)*mE_a + m2pE_a - bX_a*R_a*mE_a + uX_a*cE_a;
    dcE_a = - lambda_a*cE_a + bX_a*R_a*mE_a - uX_a*cE_a - m2pE_a;
    dpE_a = m2pE_a - lambda_a*pE_a - pE2oo_a;
    dmH_a = g2mH_a - (lambda_a + dymX_a)*mH_a + m2pH_a - bX_a*R_a*mH_a + uX_a*cH_a;
    dcH_a = - lambda_a*cH_a + bX_a*R_a*mH_a - uX_a*cH_a - m2pH_a;
    dpH_a = m2pH_a - lambda_a*pH_a;
    dmR_a = g2mR_a - (lambda_a + dymX_a)*mR_a + m2xR_a - bX_a*R_a*mR_a + uX_a*cR_a;
    dcR_a = - lambda_a*cR_a + bX_a*R_a*mR_a - uX_a*cR_a - m2xR_a;
    dxR_a = m2xR_a - lambda_a*pR_a - brho_a*pR_a*rr_a + urho_a*R_a;
    drr_a = g2rr_a - lambda_a*rr_a - brho_a*pR_a*rr_a + urho_a*R_a;
    dpR_a = brho_a*pR_a*rr_a - urho_a*R_a - lambda_a*R_a \
        + m2pT_a - bX_a*R_a*mT_a + uX_a*cT_a \
        + m2pE_a - bX_a*R_a*mE_a + uX_a*cE_a \
        + m2pH_a - bX_a*R_a*mH_a + uX_a*cH_a \
        + m2xR_a - bX_a*R_a*mR_a + uX_a*cR_a \
        + m2pI_a - bI_a*R_a*mI_a + uX_a*cI_a \
        + m2pU_a - bU_a*R_a*mU_a + uX_a*cU_a \
        + m2pP_a - bP_a*R_a*mP_a + uX_a*cP_a \
        + m2pA_a - bA*R_a*mA_a + uX_a*cA_a;
    
    # ===== Strain B cellular model ==========================================
    ## --- Host metabolism ODEs ------------------------------------------------
    diS_b = xS2iS_b - iS2iE_b - lambda_b*iS_b - dphiQB_b*iS2iQB_b;
    diE_b = phie_b*iS2iE_b - lambda_b*iE_b - nR_b*m2xR_b - nX_b*m2pT_b - nX_b*m2pE_b - nX_b*m2pH_b - nB*m2pB_b \
       - nI_b*m2pI_b - nU_b*m2pU_b - nP_b*m2pP_b;
    
    ## --- Host proteome ODEs --------------------------------------------------
    dmT_b = g2mT_b - (lambda_b + dymX_b)*mT_b + m2pT_b - bX_b*R_b*mT_b + uX_b*cT_b;
    dcT_b = - lambda_b*cT_b + bX_b*R_b*mT_b - uX_b*cT_b - m2pT_b;
    dpT_b = m2pT_b - lambda_b*pT_b;
    dmE_b = g2mE_b - (lambda_b + dymX_b)*mE_b + m2pE_b - bX_b*R_b*mE_b + uX_b*cE_b;
    dcE_b = - lambda_b*cE_b + bX_b*R_b*mE_b - uX_b*cE_b - m2pE_b;
    dpE_b = m2pE_b - lambda_b*pE_b - pE2oo_b;
    dmH_b = g2mH_b - (lambda_b + dymX_b)*mH_b + m2pH_b - bX_b*R_b*mH_b + uX_b*cH_b;
    dcH_b = - lambda_b*cH_b + bX_b*R_b*mH_b - uX_b*cH_b - m2pH_b;
    dpH_b = m2pH_b - lambda_b*pH_b;
    dmR_b = g2mR_b - (lambda_b + dymX_b)*mR_b + m2xR_b - bX_b*R_b*mR_b + uX_b*cR_b;
    dcR_b = - lambda_b*cR_b + bX_b*R_b*mR_b - uX_b*cR_b - m2xR_b;
    dxR_b = m2xR_b - lambda_b*pR_b - brho_b*pR_b*rr_b + urho_b*R_b;
    drr_b = g2rr_b - lambda_b*rr_b - brho_b*pR_b*rr_b + urho_b*R_b;
    dpR_b = brho_b*pR_b*rr_b - urho_b*R_b - lambda_b*R_b \
        + m2pT_b - bX_b*R_b*mT_b + uX_b*cT_b \
        + m2pE_b - bX_b*R_b*mE_b + uX_b*cE_b \
        + m2pH_b - bX_b*R_b*mH_b + uX_b*cH_b \
        + m2xR_b - bX_b*R_b*mR_b + uX_b*cR_b \
        + m2pI_b - bI_b*R_b*mI_b + uX_b*cI_b \
        + m2pU_b - bU_b*R_b*mU_b + uX_b*cU_b \
        + m2pP_b - bP_b*R_b*mP_b + uX_b*cP_b \
        + m2pB_b - bB*R_b*mB_b + uX_b*cB_b;

    ## ===== STRAIN A QS SYSTEM ODEs ==========================================
    ### --- LuxI ----------------------------------------------------------------
    dmI_a = g2mI_a - (lambda_a + dymX_a)*mI_a + m2pI_a - bI_a*R_a*mI_a + uX_a*cI_a;
    dcI_a = - lambda_a*cI_a + bI_a*R_a*mI_a - uX_a*cI_a - m2pI_a;
    dpI_a = m2pI_a - lambda_a*pI_a;
    
    ### --- LuxR ----------------------------------------------------------------
    dmU_a = g2mU_a - (lambda_a + dymX_a)*mU_a + m2pU_a - bU_a*R_a*mU_a + uX_a*cU_a;
    dcU_a = - lambda_a*cU_a + bU_a*R_a*mU_a - uX_a*cU_a - m2pU_a;
    dpU_a = m2pU_a - lambda_a*pU_a - bQU_a*iQB_a*pU_a + uQU_a*pUa_a;
    dpUa_a = - lambda_a*pUa_a + bQU_a*iQB_a*pU_a - uQU_a*pUa_a;
    
    ### --- Protease ------------------------------------------------------------
    dmP_a = g2mP_a - (lambda_a + dymX_a)*mP_a + m2pP_a - bP_a*R_a*mP_a + uX_a*cP_a;
    dcP_a = - lambda_a*cP_a + bP_a*R_a*mP_a - uX_a*cP_a - m2pP_a;
    dpP_a = m2pP_a - lambda_a*pP_a;
    
    ### --- QS molecules --------------------------------------------------------
    diQA_a = phiQA_a*iS2iQA_a - lambda_a*iQA_a - iQA2xQA_a;
    diQB_a = xQB2iQB_a - lambda_a*iQB_a - bQU_a*iQB_a*pU_a + uQU_a*pUa_a;
    
    ## ===== STRAIN B QS SYSTEM ODEs ==========================================
    ### --- LuxI ----------------------------------------------------------------
    dmI_b = g2mI_b - (lambda_b + dymX_b)*mI_b + m2pI_b - bI_b*R_b*mI_b + uX_b*cI_b;
    dcI_b = - lambda_b*cI_b + bI_b*R_b*mI_b - uX_b*cI_b - m2pI_b;
    dpI_b = m2pI_b - lambda_b*pI_b;
    
    ### --- LuxR ----------------------------------------------------------------
    dmU_b = g2mU_b - (lambda_b + dymX_b)*mU_b + m2pU_b - bU_b*R_b*mU_b + uX_b*cU_b;
    dcU_b = - lambda_b*cU_b + bU_b*R_b*mU_b - uX_b*cU_b - m2pU_b;
    dpU_b = m2pU_b - lambda_b*pU_b - bQU_b*iQA_b*pU_b + uQU_b*pUa_b;
    dpUa_b = - lambda_b*pUa_b + bQU_b*iQA_b*pU_b - uQU_b*pUa_b;
    
    ### --- Protease ------------------------------------------------------------
    dmP_b = g2mP_b - (lambda_b + dymX_b)*mP_b + m2pP_b - bP_b*R_b*mP_b + uX_b*cP_b;
    dcP_b = - lambda_b*cP_b + bP_b*R_b*mP_b - uX_b*cP_b - m2pP_b;
    dpP_b = m2pP_b - lambda_b*pP_b;
    
    ### --- QS molecules --------------------------------------------------------
    diQA_b = xQA2iQA_b - lambda_b*iQA_b - bQU_b*iQA_b*pU_b + uQU_b*pUa_b;
    diQB_b = phiQB_b*iS2iQB_b - lambda_b*iQB_b - iQB2xQB_b;

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
    
    ## --- Culture ODEs --------------------------------------------------------
    dxS = - xS2iS_a*N_a - xS2iS_b*N_b;
    dxA = iA2xA_a*N_a - xA2iA_b*N_b;
    dxB = iB2xB_b*N_b;
    dN_a = lambda_a*N_a;
    dN_b = lambda_b*N_b;
    dxQA = iQA2xQA_a*N_a - xQA2iQA_b*N_b;
    dxQB = iQB2xQB_b*N_b - xQB2iQB_a*N_a;

    # ===== Return ===========================================================
    dY_by_dt = jnp.array([dxS,dxA,dxB,dN_a,dN_b,
                          diS_a,diE_a,
                          dmT_a,dcT_a,dpT_a,dmE_a,dcE_a,dpE_a,
                          dmH_a,dcH_a,dpH_a,dmR_a,dcR_a,dxR_a,drr_a,dpR_a,
                          dmA_a,dcA_a,dpA_a,diA_a,
                          diS_b,diE_b,
                          dmT_b,dcT_b,dpT_b,dmE_b,dcE_b,dpE_b,
                          dmH_b,dcH_b,dpH_b,dmR_b,dcR_b,dxR_b,drr_b,dpR_b,
                          dmB_b,dcB_b,dpB_b,
                          diA_b,diB_b,
                          dmI_a,dcI_a,dpI_a,dmU_a,dcU_a,dpU_a,dpUa_a,dmP_a, dcP_a, dpP_a,
                          diQA_a,diQB_a,
                          dmI_b,dcI_b,dpI_b,dmU_b,dcU_b,dpU_b,dpUa_b,dmP_b, dcP_b,dpP_b,
                          diQA_b,diQB_b,
                          dxQA,dxQB])

    dY_by_dt = jnp.where(Y<=0, jnp.maximum(dY_by_dt, 0), dY_by_dt)
    population    = jnp.array([N_a,N_b]);
    growth_rate   = jnp.array([lambda_a,lambda_b]);
    extracellular_metabolism = jnp.array([xS, xA, xB, xQA, xQB]);
    intracellular_metabolism = jnp.array([iA_a, iA_b, 0, iB_b]);
    pathway_gene = jnp.array([mA_a, cA_a, pA_a, mB_b, cB_b, pB_b]);
    
    return (dY_by_dt, population, growth_rate, extracellular_metabolism, 
            intracellular_metabolism, pathway_gene)

@equinox.filter_jit
def twostrainsQSDinhlinxpODE_wrapped(T, Y, args):
    dy_dt, _, _, _, _, _ = twostrainsQSDinhlinxpODE(T, Y, args)
    return dy_dt

W_twostrainsQSDinhlinxpODE = lambda T, Y, arg1, arg2: twostrainsQSDinhlinxpODE_wrapped(T, Y, (arg1, arg2))

# ===== SIMULATE TWO STRAIN MODEL WITH QS-MEDIATED BURDEN TUNING ========================================
@equinox.filter_jit
def twostrainsQSBoolinxpODE(T, Y, args):
# ======================================================================= #
#  --- Reference -----------------------------------------------------------
#  Boo, Mehta, Ledesma-Amaro, and Stan 2023 
#  https://doi.org/10.1101/2023.05.15.540816
#
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
    iS_a = Y[5]; iE_a  = Y[6];                                                 # internal substrate, internal energy
    mT_a = Y[7]; cT_a  = Y[8]; pT_a  = Y[9];                                   # transporter mRNA, translation complex, protein
    mE_a = Y[10]; cE_a = Y[11]; pE_a = Y[12];                                  # enzyme mRNA, translation complex, protein
    mH_a = Y[13]; cH_a = Y[14]; pH_a = Y[15];                                  # host q-protein mRNA, translation complex, protein
    mR_a = Y[16]; cR_a = Y[17]; pR_a = Y[18];                                  # r-protein mRNA, translation complex, protein
    rr_a = Y[19]; R_a  = Y[20];                                                # rRNA, free ribosomes 
    mA_a = Y[21]; cA_a = Y[22]; pA_a = Y[23]; iA_a = Y[24];                    # heterologous gene A mRNA, translation complex, protein, intracellular product A
    
    ### --- Strain B ------------------------------------------------------------
    iS_b = Y[25]; iE_b = Y[26];                                                # internal substrate, internal energy
    mT_b = Y[27]; cT_b = Y[28]; pT_b = Y[29];                                  # transporter mRNA, translation complex, protein
    mE_b = Y[30]; cE_b = Y[31]; pE_b = Y[32];                                  # enzyme mRNA, translation complex, protein
    mH_b = Y[33]; cH_b = Y[34]; pH_b = Y[35];                                  # host q-protein mRNA, translation complex, protein
    mR_b = Y[36]; cR_b = Y[37]; pR_b = Y[38];                                  # r-protein mRNA, translation complex, protein
    rr_b = Y[39]; R_b  = Y[40];                                                # rRNA, free ribosomes 
    mB_b = Y[41]; cB_b = Y[42]; pB_b = Y[43]; iA_b = Y[44]; iB_b = Y[45];      # heterologous gene B mRNA, translation complex, protein, intracellular product B
    
    ### --- Strain A QS system --------------------------------------------------
    mI_a = Y[46]; cI_a = Y[47]; pI_a = Y[48];                                  # LuxI mRNA, translation complex, protein
    mU_a = Y[49]; cU_a = Y[50]; pU_a = Y[51]; pUa_a = Y[52];                   # LuxR mRNA, translation complex, inactive protein, active (qs-bound) protein 
    rS_a = Y[53];                                                              # rSNA inhibited mA_a
    iQA_a = Y[54]; iQB_a = Y[55];                                              # QS molecules inside strain A

    ### --- Strain B QS system --------------------------------------------------
    mI_b = Y[56]; cI_b = Y[57]; pI_b = Y[58];                                  # LuxI mRNA, translation complex, protein
    mU_b = Y[59]; cU_b = Y[60]; pU_b = Y[61]; pUa_b = Y[62];                   # LuxR mRNA, translation complex, inactive protein, active [qs-bound] protein 
    rS_b = Y[63];                                                              # rSNA inhibited mB_b
    iQA_b = Y[64]; iQB_b = Y[65];                                              # QS molecules inside strain B

    ### --- Extracellular QS molecules ------------------------------------------
    xQA = Y[66]; xQB = Y[67];                                                  # QS molecules

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

    ### --- New parameters ------------------------------------------------------
    wI_a    = hPR[48]; wU_a     = hPR[49]; wS_a = hPR[50];                     # transcription rates luxI, luxR and sRNA
    nI_a    = hPR[51]; nU_a     = hPR[52];                                     # protein lengths in amino acids
    bI_a    = hPR[53]; bU_a     = hPR[54];                                     # RBS binding rate
    bQU_a   = hPR[55]; uQU_a    = hPR[56];                                     # QS-LuxR binding rates
    kU_a    = hPR[57]; hU_a     = hPR[58];                                     # LuxR hill parameters
    phiQA_a = hPR[59]; dphiQA_a = hPR[60];                                     # QS
    vEiQA_a = hPR[61]; kEiQA_a  = hPR[62];                                     #
    vXiQA_a = hPR[63]; vIiQB_a  = hPR[64];                                     # iQA export rate, iQB import rate
    dyrS_a  = hPR[65]; bRM_a    = hPR[66];                                     # sRNA decay rate, sRNA:mRNA binding rate
	
    wI_b    = hPR[67]; wU_b     = hPR[68]; wS_b = hPR[69];                     # transcription rates luxI, luxR and sRNA
    nI_b    = hPR[70]; nU_b     = hPR[71];                                     # protein lengths in amino acids
    bI_b    = hPR[72]; bU_b     = hPR[73];                                     # RBS binding rate
    bQU_b   = hPR[74]; uQU_b    = hPR[75];                                     # QS-LuxR binding rates
    kU_b    = hPR[76]; hU_b     = hPR[77];                                     # LuxR hill parameters
    phiQB_b = hPR[78]; dphiQB_b = hPR[79];                                     # QS
    vEiQB_b = hPR[80]; kEiQB_b  = hPR[81];                                     #
    vXiQB_b = hPR[82]; vIiQA_b  = hPR[83];                                     # iQA export rate, iQB import rate
    dyrS_b  = hPR[84]; bRM_b    = hPR[85];                                     # sRNA decay rate, sRNA:mRNA binding rate

    # ==== Circuit parameters ================================================
    wA  = cPR[0]; nA   = cPR[1]; bA    = cPR[2];                               # Protein A transcription rate, protein length, RBS strength
    vAe = cPR[3]; kAe  = cPR[4]; vIxA  = cPR[5]; vXiA = cPR[6];                # Protein A kcat, kM, import rate, export rate
    wB  = cPR[7]; nB   = cPR[8]; bB    = cPR[9];                               # Protein B transcription rate, protein length, RBS strength
    vBe = cPR[10]; kBe = cPR[11]; vXiB = cPR[12];                            # Protein B kcat, kM, import rate, export rate
    
    # ===== Calculate Strain A rates =========================================
    ## --- Global translation rates --------------------------------------------
    gammaX_a = (maxG_a*iE_a)/(kG_a + iE_a);
    
    ## --- Growth rates ---------------------------------------------------------
    lambda_a = (1/M0_a)*gammaX_a*(cT_a + cE_a + cH_a + cR_a + cA_a + cI_a + cU_a);
    
    ## --- Transcription rates -------------------------------------------------
    g2mT_a = (wT_a*iE_a)/(oX_a + iE_a);
    g2mE_a = (wE_a*iE_a)/(oX_a + iE_a);
    g2mH_a = ((wH_a*iE_a)/(oX_a + iE_a))*(1/(1+(pH_a/kH_a)**hH_a));
    g2mR_a = (wR_a*iE_a)/(oR_a + iE_a);
    g2rr_a = (wr_a*iE_a)/(oR_a + iE_a);
    g2mA_a = (wA*iE_a)/(oX_a + iE_a);
    g2mI_a = (wI_a*iE_a)/(oX_a + iE_a);
    g2mU_a = (wU_a*iE_a)/(oX_a + iE_a);
    g2rS_a = ((wS_a*iE_a)/(oX_a + iE_a))*((pUa_a**hU_a)/((kU_a**hU_a) + (pUa_a**hU_a)));

    ## --- Translation rates ---------------------------------------------------
    m2pT_a = (gammaX_a/nX_a)*cT_a;
    m2pE_a = (gammaX_a/nX_a)*cE_a;
    m2pH_a = (gammaX_a/nX_a)*cH_a;
    m2xR_a = (gammaX_a/nR_a)*cR_a;
    m2pA_a = (gammaX_a/nA)*cA_a;
    m2pI_a = (gammaX_a/nI_a)*cI_a;
    m2pU_a = (gammaX_a/nU_a)*cU_a;
    
    ## ---- Metabolic rates ----------------------------------------------------
    xS2iS_a = (vIxS_a*xS*pT_a)/(kIxS_a + xS);
    iS2iE_a = (vEe_a*iS_a*pE_a)/(kEe_a + iS_a);
    iS2iQA_a  = (vEiQA_a*iS_a*pI_a)/(kEiQA_a + iS_a);
    iQA2xQA_a = (vXiQA_a*iQA_a);
    xQB2iQB_a = (vIiQB_a*xQB);

    # ---- Pathway rates ------------------------------------------------------
    # xS --[-- iS --> iA --]--> xA
    iS2iA_a = (vAe*iS_a*pA_a)/(kAe + iS_a); # iS ---> iA
    iA2xA_a = vXiA*iA_a; # iA -]-> xA
    
    # ===== Calculate Strain B rates =========================================
    ## --- Global translation rates --------------------------------------------
    gammaX_b = (maxG_b*iE_b)/(kG_b + iE_b);
    
    ## --- Growth rates ---------------------------------------------------------
    lambda_b = (1/M0_b)*gammaX_b*(cT_b + cE_b + cH_b + cR_b + cB_b + cI_b + cU_b);
    
    ## --- Transcription rates -------------------------------------------------
    g2mT_b = (wT_b*iE_b)/(oX_b + iE_b);
    g2mE_b = (wE_b*iE_b)/(oX_b + iE_b);
    g2mH_b = ((wH_b*iE_b)/(oX_b + iE_b))*(1/(1+(pH_b/kH_b)**hH_b));
    g2mR_b = (wR_b*iE_b)/(oR_b + iE_b);
    g2rr_b = (wr_b*iE_b)/(oR_b + iE_b);
    g2mB_b = (wB*iE_b)/(oX_b + iE_b);
    g2mI_b = (wI_b*iE_b)/(oX_b + iE_b);
    g2mU_b = (wU_b*iE_b)/(oX_b + iE_b);
    g2rS_b = ((wS_b*iE_b)/(oX_b + iE_b))*(((pUa_b/kU_b)**hU_b)/(1 + ((pUa_b/kU_b)**hU_b)));
    
    ## --- Translation rates ---------------------------------------------------
    m2pT_b = (gammaX_b/nX_b)*cT_b;
    m2pE_b = (gammaX_b/nX_b)*cE_b;
    m2pH_b = (gammaX_b/nX_b)*cH_b;
    m2xR_b = (gammaX_b/nR_b)*cR_b;
    m2pB_b = (gammaX_b/nB)*cB_b;
    m2pI_b = (gammaX_b/nI_b)*cI_b;
    m2pU_b = (gammaX_b/nU_b)*cU_b;
    
    ## ---- Metabolic rates ----------------------------------------------------
    xS2iS_b = (vIxS_b*xS*pT_b)/(kIxS_b + xS);
    iS2iE_b = (vEe_b*iS_b*pE_b)/(kEe_b + iS_b);
    iS2iQB_b  = (vEiQB_b*iS_b*pI_b)/(kEiQB_b + iS_b);
    iQB2xQB_b = (vXiQB_b*iQB_b);
    xQA2iQA_b = (vIiQA_b*xQA);
    
    ## --- Pathway rates -------------------------------------------------------
    # xA --[--> iA --> iB --]--> xB
    xA2iA_b = vIxA*xA; # xA -[-> iA
    iA2iB_b = (vBe*iA_b*pB_b)/(kBe + iA_b); # iA ---> iB
    iB2xB_b = vXiB*iB_b; # iB -]-> xB
    
    # ===== Strain A cellular model ==========================================
    ## --- Host metabolism ODEs ------------------------------------------------
    diS_a = xS2iS_a - iS2iE_a - iS2iA_a - lambda_a*iS_a - dphiQA_a*iS2iQA_a;
    diE_a = phie_a*iS2iE_a - lambda_a*iE_a - nR_a*m2xR_a - nX_a*m2pT_a - nX_a*m2pE_a - nX_a*m2pH_a - nA*m2pA_a \
        - nI_a*m2pI_a - nU_a*m2pU_a;
    
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
    dmR_a = g2mR_a - (lambda_a + dymX_a)*mR_a + m2xR_a - bX_a*R_a*mR_a + uX_a*cR_a;
    dcR_a = - lambda_a*cR_a + bX_a*R_a*mR_a - uX_a*cR_a - m2xR_a;
    dpR_a = m2xR_a - lambda_a*pR_a - brho_a*pR_a*rr_a + urho_a*R_a;
    drr_a = g2rr_a - lambda_a*rr_a - brho_a*pR_a*rr_a + urho_a*R_a;
    dR_a = brho_a*pR_a*rr_a - urho_a*R_a - lambda_a*R_a \
        + m2pT_a - bX_a*R_a*mT_a + uX_a*cT_a \
        + m2pE_a - bX_a*R_a*mE_a + uX_a*cE_a \
        + m2pH_a - bX_a*R_a*mH_a + uX_a*cH_a \
        + m2xR_a - bX_a*R_a*mR_a + uX_a*cR_a \
        + m2pI_a - bI_a*R_a*mI_a + uX_a*cI_a \
        + m2pU_a - bU_a*R_a*mU_a + uX_a*cU_a \
        + m2pA_a - bA*R_a*mA_a + uX_a*cA_a;
    
    # ===== Strain B cellular model ==========================================
    ## --- Host metabolism ODEs ------------------------------------------------
    diS_b = xS2iS_b - iS2iE_b - lambda_b*iS_b - dphiQB_b*iS2iQB_b;
    diE_b = phie_b*iS2iE_b - lambda_b*iE_b - nR_b*m2xR_b - nX_b*m2pT_b - nX_b*m2pE_b - nX_b*m2pH_b - nB*m2pB_b \
       - nI_b*m2pI_b - nU_b*m2pU_b;
    
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
    dmR_b = g2mR_b - (lambda_b + dymX_b)*mR_b + m2xR_b - bX_b*R_b*mR_b + uX_b*cR_b;
    dcR_b = - lambda_b*cR_b + bX_b*R_b*mR_b - uX_b*cR_b - m2xR_b;
    dpR_b = m2xR_b - lambda_b*pR_b - brho_b*pR_b*rr_b + urho_b*R_b;
    drr_b = g2rr_b - lambda_b*rr_b - brho_b*pR_b*rr_b + urho_b*R_b;
    dR_b = brho_b*pR_b*rr_b - urho_b*R_b - lambda_b*R_b \
        + m2pT_b - bX_b*R_b*mT_b + uX_b*cT_b \
        + m2pE_b - bX_b*R_b*mE_b + uX_b*cE_b \
        + m2pH_b - bX_b*R_b*mH_b + uX_b*cH_b \
        + m2xR_b - bX_b*R_b*mR_b + uX_b*cR_b \
        + m2pI_b - bI_b*R_b*mI_b + uX_b*cI_b \
        + m2pU_b - bU_b*R_b*mU_b + uX_b*cU_b \
        + m2pB_b - bB*R_b*mB_b + uX_b*cB_b;

    ## ===== STRAIN A QS SYSTEM ODEs ==========================================
    ### --- LuxI ----------------------------------------------------------------
    dmI_a = g2mI_a - (lambda_a + dymX_a)*mI_a + m2pI_a - bI_a*R_a*mI_a + uX_a*cI_a;
    dcI_a = - lambda_a*cI_a + bI_a*R_a*mI_a - uX_a*cI_a - m2pI_a;
    dpI_a = m2pI_a - lambda_a*pI_a;
    
    ### --- LuxR ----------------------------------------------------------------
    dmU_a = g2mU_a - (lambda_a + dymX_a)*mU_a + m2pU_a - bU_a*R_a*mU_a + uX_a*cU_a;
    dcU_a = - lambda_a*cU_a + bU_a*R_a*mU_a - uX_a*cU_a - m2pU_a;
    dpU_a = m2pU_a - lambda_a*pU_a - bQU_a*iQB_a*pU_a + uQU_a*pUa_a;
    dpUa_a = - lambda_a*pUa_a + bQU_a*iQB_a*pU_a - uQU_a*pUa_a;
    
    ### --- Protease ------------------------------------------------------------
    drS_a = g2rS_a - (lambda_a + dyrS_a)*rS_a - bRM_a*rS_a*mA_a;
    
    ### --- QS molecules --------------------------------------------------------
    diQA_a = phiQA_a*iS2iQA_a - lambda_a*iQA_a - iQA2xQA_a;
    diQB_a = xQB2iQB_a - lambda_a*iQB_a - bQU_a*iQB_a*pU_a + uQU_a*pUa_a;
    
    ## ===== STRAIN B QS SYSTEM ODEs ==========================================
    ### --- LuxI ----------------------------------------------------------------
    dmI_b = g2mI_b - (lambda_b + dymX_b)*mI_b + m2pI_b - bI_b*R_b*mI_b + uX_b*cI_b;
    dcI_b = - lambda_b*cI_b + bI_b*R_b*mI_b - uX_b*cI_b - m2pI_b;
    dpI_b = m2pI_b - lambda_b*pI_b;
    
    ### --- LuxR ----------------------------------------------------------------
    dmU_b = g2mU_b - (lambda_b + dymX_b)*mU_b + m2pU_b - bU_b*R_b*mU_b + uX_b*cU_b;
    dcU_b = - lambda_b*cU_b + bU_b*R_b*mU_b - uX_b*cU_b - m2pU_b;
    dpU_b = m2pU_b - lambda_b*pU_b - bQU_b*iQA_b*pU_b + uQU_b*pUa_b;
    dpUa_b = - lambda_b*pUa_b + bQU_b*iQA_b*pU_b - uQU_b*pUa_b;
    
    ### --- Protease ------------------------------------------------------------
    drS_b = g2rS_b - (lambda_b + dyrS_b)*rS_b - bRM_b*rS_b*mB_b;
    
    ### --- QS molecules --------------------------------------------------------
    diQA_b = xQA2iQA_b - lambda_b*iQA_b - bQU_b*iQA_b*pU_b + uQU_b*pUa_b;
    diQB_b = phiQB_b*iS2iQB_b - lambda_b*iQB_b - iQB2xQB_b;

    # ===== Pathway ODEs =====================================================
    ## --- Strain A circuit and pathway ODEs -----------------------------------
    dmA_a = g2mA_a - (lambda_a + dymX_a)*mA_a + m2pA_a - bA*R_a*mA_a + uX_a*cA_a - bRM_a*rS_a*mA_a;
    dcA_a = - lambda_a*cA_a + bA*R_a*mA_a - uX_a*cA_a - m2pA_a;
    dpA_a = m2pA_a - lambda_a*pA_a;
    diA_a = iS2iA_a - iA2xA_a - lambda_a*iA_a;
    
    ## --- Strain B Circuit and pathway ODEs -----------------------------------
    dmB_b = g2mB_b - (lambda_b + dymX_b)*mB_b + m2pB_b - bB*R_b*mB_b + uX_b*cB_b - bRM_b*rS_b*mB_b;
    dcB_b = - lambda_b*cB_b + bB*R_b*mB_b - uX_b*cB_b - m2pB_b;
    dpB_b = m2pB_b - lambda_b*pB_b;
    diA_b = xA2iA_b - iA2iB_b - lambda_b*iA_b;
    diB_b = iA2iB_b - iB2xB_b - lambda_b*iB_b;
    
    ## --- Culture ODEs --------------------------------------------------------
    dxS = - xS2iS_a*N_a - xS2iS_b*N_b;
    dxA = iA2xA_a*N_a - xA2iA_b*N_b;
    dxB = iB2xB_b*N_b;
    dN_a = lambda_a*N_a;
    dN_b = lambda_b*N_b;
    dxQA = iQA2xQA_a*N_a - xQA2iQA_b*N_b;
    dxQB = iQB2xQB_b*N_b - xQB2iQB_a*N_a;

    # ===== Return ===========================================================
    dY_by_dt = jnp.array([dxS,dxA,dxB,dN_a,dN_b,
                          diS_a,diE_a,
                          dmT_a,dcT_a,dpT_a,dmE_a,dcE_a,dpE_a,
                          dmH_a,dcH_a,dpH_a,dmR_a,dcR_a,dpR_a,drr_a,dR_a,
                          dmA_a,dcA_a,dpA_a,diA_a,
                          diS_b,diE_b,
                          dmT_b,dcT_b,dpT_b,dmE_b,dcE_b,dpE_b,
                          dmH_b,dcH_b,dpH_b,dmR_b,dcR_b,dpR_b,drr_b,dR_b,
                          dmB_b,dcB_b,dpB_b,diA_b,diB_b,
                          dmI_a,dcI_a,dpI_a,dmU_a,dcU_a,dpU_a,dpUa_a,drS_a,
                          diQA_a,diQB_a,
                          dmI_b,dcI_b,dpI_b,dmU_b,dcU_b,dpU_b,dpUa_b,drS_b,
                          diQA_b,diQB_b,
                          dxQA,dxQB])

    dY_by_dt = jnp.where(Y<=0, jnp.maximum(dY_by_dt, 0), dY_by_dt)
    population    = jnp.array([N_a,N_b]);
    growth_rate   = jnp.array([lambda_a,lambda_b]);
    extracellular_metabolism = jnp.array([xS, xA, xB, xQA, xQB]);
    intracellular_metabolism = jnp.array([iA_a, iA_b, 0, iB_b]);
    pathway_gene = jnp.array([mA_a, cA_a, pA_a, mB_b, cB_b, pB_b]);
    
    return (dY_by_dt, population, growth_rate, extracellular_metabolism, 
            intracellular_metabolism, pathway_gene)

@equinox.filter_jit
def twostrainsQSBoolinxpODE_wrapped(T, Y, args):
    dy_dt, _, _, _, _, _ = twostrainsQSBoolinxpODE(T, Y, args)
    return dy_dt

W_twostrainsQSBoolinxpODE = lambda T, Y, arg1, arg2: twostrainsQSBoolinxpODE_wrapped(T, Y, (arg1, arg2))

@equinox.filter_jit
def explicit_term(t, y, args):
    return 0.0