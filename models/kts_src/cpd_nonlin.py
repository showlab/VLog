import numpy as np
# from scipy import weave

def calc_scatters(K):
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n+1, n+1))
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1) # TODO: use the fact that K - symmetric

    scatters = np.zeros((n, n))
    
#     code = r"""
#     for (int i = 0; i < n; i++) {
#         for (int j = i; j < n; j++) {
#             scatters(i,j) = K1(j+1)-K1(i) - (K2(j+1,j+1)+K2(i,i)-K2(j+1,i)-K2(i,j+1))/(j-i+1);
#         }
#     }
#     """
#     weave.inline(code, ['K1','K2','scatters','n'], global_dict = \
#         {'K1':K1, 'K2':K2, 'scatters':scatters, 'n':n}, type_converters=weave.converters.blitz)
    
    for i in range(n):
        for j in range(i, n):
            scatters[i,j] = K1[j+1] - K1[i] - (K2[j+1,j+1]+K2[i,i]-K2[j+1,i]-K2[i,j+1])/(j-i+1)
    return scatters

def cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True, verbose=False,
    out_scatters=None):
    """ Change point detection with dynamic programming
    K - square kernel matrix 
    ncp - number of change points to detect (ncp >= 0)
    lmin - minimal length of a segment
    lmax - maximal length of a segment
    backtrack - when False - only evaluate objective scores (to save memory)
    
    Returns: (cps, obj)
        cps - detected array of change points: mean is thought to be constant on [ cps[i], cps[i+1] )    
        obj_vals - values of the objective function for 0..m changepoints
        
    """   
    m = int(ncp)  # prevent numpy.int64
    
    (n, n1) = K.shape
    assert(n == n1), "Kernel matrix awaited."    
    
    assert(n >= (m + 1)*lmin)
    assert(n <= (m + 1)*lmax)
    assert(lmax >= lmin >= 1)
    
    # if verbose:
        # print("Precomputing scatters...")
    J = calc_scatters(K)
    
    if out_scatters != None:
        out_scatters[0] = J

    # if verbose:
        # print("Inferring best change points...")
    I = 1e101*np.ones((m+1, n+1))
    I[0, lmin:lmax] = J[0, lmin-1:lmax-1]

    if backtrack:
        p = np.zeros((m+1, n+1), dtype=int)
    else:
        p = np.zeros((1,1), dtype=int)
        
#     code = r"""
#     #define max(x,y) ((x)>(y)?(x):(y))
#     for (int k=1; k<m+1; k++) {
#         for (int l=(k+1)*lmin; l<n+1; l++) {
#             I(k, l) = 1e100; //nearly infinity
#             for (int t=max(k*lmin,l-lmax); t<l-lmin+1; t++) {
#                 double c = I(k-1, t) + J(t, l-1);
#                 if (c < I(k, l)) {
#                     I(k, l) = c;
#                     if (backtrack == 1) {
#                         p(k, l) = t;
#                     }
#                 }
#             }
#         }
#     }
#     """

#     weave.inline(code, ['m','n','p','I', 'J', 'lmin', 'lmax', 'backtrack'], \
#         global_dict={'m':m, 'n':n, 'p':p, 'I':I, 'J':J, \
#         'lmin':lmin, 'lmax':lmax, 'backtrack': int(1) if backtrack else int(0)},
#         type_converters=weave.converters.blitz)
    
    for k in range(1, m+1):
        for l in range((k+1)*lmin, n+1):
            I[k, l] = 1e100
            for t in range(max(k*lmin,l-lmax), l-lmin+1):
                c = I[k-1, t] + J[t, l-1]
                if (c < I[k, l]):
                    I[k, l] = c
                    if (backtrack == 1):
                        p[k, l] = t
    
    
    # Collect change points
    cps = np.zeros(m, dtype=int)
    
    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k-1] = p[k, cur]
            cur = cps[k-1]

    scores = I[:, n].copy() 
    scores[scores > 1e99] = np.inf
    return cps, scores
    

