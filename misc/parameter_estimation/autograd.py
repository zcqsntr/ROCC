import autograd



def sdot2(S, t, Cin, A, params, num_species): # X is population vector, t is time, R is intrinsic growth rate vector, C is the rate limiting nutrient vector, A is interaction matrix
    N = np.array(S[:num_species])
    C = np.array(S[num_species:2*num_species])
    C0 = np.array(S[-1])

    C0in, q, y, y3, Rmax, Km, Km3 = params

    R = monod2(C, C0, Rmax, Km, Km3)

    Cin = Cin[:num_species]

    dN = N * (R + np.matmul(A,N) - q) # q term takes account of the dilution
    dC = q*(Cin - C) - (1/y)*R*N # sometimes dC.shape is (2,2)
    dC0 = q*(C0in - C0) - sum(1/y3[i]*R[i]*N[i] for i in range(num_species))

    if dC.shape == (2,2):
        print(q,Cin.shape,C0,C,y,R,N)
    dC0 = np.array([dC0])
    sol = np.append(dN, dC)
    sol = np.append(sol, dC0)
    return tuple(sol)



def likelihood_objective(observed_dN, param_vec):
    pass


def squared_objective(observed_N, last_S, param_vec):



    return (predicted_N - actual_N)**2
