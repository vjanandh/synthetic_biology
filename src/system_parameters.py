from dataclasses import dataclass

@dataclass
class SystemParameters:
    ## --- Environmental conditions --------------------------------------------
    phie0: float = 0.5   # nutrient efficiency
    
    ## --- Host parameters -----------------------------------------------------
    vIxS: float = 726    # max. nutrient import rate
    vEe: float = 5800    # max. enzymatic rate
    kIxS: float = 1e3    # nutrient import threhold
    kEe: float = 1e3     # enzymatic threshold
	
    wT: float = 4.14     # max transport enzyme transc. rate
    wE: float = 4.14     # max metabolic enzyme transc. rate
    wH: float = 948.93   # max. q. transcription rate
    wR: float = 930      # max ribosome transc. rate
    wr: float = 3170     
    
    oX: float = 4.38     # non-ribosomal transcription threshold
    oR: float = 426.87   # ribosome transcription threshold
    
    nX: float = 300      # non-ribosome length
    nR: float = 7459     # ribosome length
    
    bX: float = 1        
    uX: float = 1        
    brho: float = 1      # ribosome binding rate
    urho: float = 1      # ribosome unbinding rate
    
    dymX: float = 0.1    # mRNA degradation rate
    
    maxG: float = 1260   # max transl. elong. rate
    kappaG: float = 7    # transl. elong. threshold
    
    M0: float = 1e8      # total cell mass
    
    kH: float = 152219   # q-autoinhibition threshold
    hH: float = 4        # q-autoinhibition Hill coeff
    
    ## --- Circuit parameters --------------------------------------------------
    r0: float = 0.5
       
    ### Strain A
    wA: float =  0
    nA: float = 600
    bA: float = 1
    vAe: float = 5800
    kAe: float = 1e3
    vIxA: float = 1
    vXiA: float = 1
    
    ### Strain B
    wB: float =  0
    nB: float = 1500
    bB: float = 1
    vBe: float = 58
    kBe: float = 1e9
    vXiB: float = 1

    ## --- Initial conditions --------------------------------------------------
    initxS0: float = 1e4 # external nutrient
    xS0: float = 1e15
    xA0: float = 0
    xB0: float = 0
    iS0: float = 1e3
    eE0: float = 1e3
    pT0: float = 10
    pE0: float = 10
    pH0: float = 10
    pR0: float = 10
    rr0: float = 10
    R0: float = 10
    pA0: float = 0
    pB0: float = 0
    N0: float = 1e3

    ## Simulation conditions
    tmax: float = 7*24*60
    runintmax: float = int(1e6)

@dataclass
class TwostrainsXfeedParameters:
    vXiEA: float = 1
    vIiEB: float = 1
    vIxEA: float = 1
    vXiEB: float = 1

@dataclass
class TwostrainsQSDecayParameters:
    wI_a: float = 1
    wU_a: float = 1
    wP_a: float = 1
    
    nI_a: float = 300
    nU_a: float = 300
    nP_a: float = 300
    
    bI_a: float = 1
    bU_a: float = 1
    bP_a: float = 1
    
    bQU_a: float = 1
    uQU_a: float = 1
    
    kU_a: float = 1e0
    hU_a: float = 2
    
    phiQA_a: float = 1
    dphiQA_a: float = 0
    
    vEiQA_a: float = 58 
    kEiQA_a: float = 1000
    
    vXiQA_a: float = 1
    vIiQB_a: float = 1
    
    vP_a: float = 0.1
    kP_a: float = 1e3
    
    wI_b: float = 1
    wU_b: float = 1
    wP_b: float = 1
    
    nI_b: float = 300
    nU_b: float = 300
    nP_b: float = 300
    
    bI_b: float = 1
    bU_b: float = 1
    bP_b: float = 1
    
    bQU_b: float = 1
    uQU_b: float = 1
    
    kU_b: float = 1e0
    hU_b: float = 2
    
    phiQB_b: float = 1
    dphiQB_b: float = 0
    
    vEiQB_b: float = 58
    kEiQB_b: float = 1000
    
    vXiQB_b: float = 1
    vIiQA_b: float = 1
    
    vP_b: float = 0.1
    kP_b: float = 1e3

@dataclass
class TwostrainsQSBurdenParameters:
    wI_a: float = 1
    wU_a: float = 1
    wS_a: float = 10
    
    nI_a: float = 300
    nU_a: float = 300
    
    bI_a: float = 1
    bU_a: float = 1
    
    bQU_a: float = 1
    uQU_a: float = 0
    
    kU_a: float = 1e0
    hU_a: float = 2
    
    phiQA_a: float = 1
    dphiQA_a: float = 0
    
    vEiQA_a: float = 58 
    kEiQA_a: float = 1000
    
    vXiQA_a: float = 1
    vIiQB_a: float = 1
    
    dyrS_a: float = 0.1
    bRM_a: float = 1
    
    wI_b: float = 1
    wU_b: float = 1
    wS_b: float = 10
    
    nI_b: float = 300
    nU_b: float = 300
    
    bI_b: float = 1
    bU_b: float = 1
    
    bQU_b: float = 1
    uQU_b: float = 0
    
    kU_b: float = 1e0
    hU_b: float = 2
    
    phiQB_b: float = 1
    dphiQB_b: float = 0
    
    vEiQB_b: float = 58
    kEiQB_b: float = 1000
    
    vXiQB_b: float = 1
    vIiQA_b: float = 1
    
    dyrS_b: float = 0.1
    bRM_b: float = 1
