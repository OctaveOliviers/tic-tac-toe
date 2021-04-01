# @Created by: OctaveOliviers
# @Created on: 2021-01-16 01:02:03
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-03-15 16:23:58


# import libraries
using LinearAlgebra


# macro Name(arg)
#     """
#     read name of a variable
#     """
#     string(arg)
# end


function normalize(args)
    #function body
end

function compute_potential(P, T, r, p, q, gam, len_epi; return_q_opt=false)
    """
    explain
    """
    # extract useful info
    num_sa,num_s = size(P)
    
    A = P*T

    diag = zeros(Float64, num_sa)
    block = zeros(Float64, num_sa, num_sa)

    pot = 0

    for l in 0:len_epi
        # [I + gam A.T + ... + gam^(L-l) A.T^(L-l)]
        mat_sum = inv(I-gam*A') * (I-gam^(len_epi+1-l)*A'^(len_epi+1-l))
        # A^l p
        Al_p = A^l * p
        # diag(p + A p + ... + A^l p)
        diag += Al_p
        # [diag(p) [I + ... + gam^L A.T^L] + ... + diag(A^L p) [I]]
        block += Diagonal(Al_p) * mat_sum
    end
    
    pot += q'*Diagonal(diag)*q /2

    pot -= r'*block'*q
    
    pot += r'*block'* Diagonal(1 ./diag)*block*r /2

    val = P'*Diagonal(1 ./diag)*block*r
    # val = - np.matmul( np.matmul(r.T, block.T)/diag, np.matmul(block, r) ) /2

    if return_q_opt
        # q_opt = Diagonal(diag)\(block*r)
        q_opt = (I-gam*T'*P')\r
        return pot, val, q_opt
    else
        return pot, val
    end
end


function step(P, T, r, p, q, gam, len_epi; return_diag=false)
    """
    explain
    """

    # extract useful info
    num_sa,num_s = size(P)
    # build system matrix
    A = P*T

    diag = zeros(Float64, num_sa)
    block = zeros(Float64, num_sa, num_sa)

    for l in 0:len_epi
        # [I + gam A.T + ... + gam^(L-l) A.T^(L-l)]
        mat_sum = inv(I-gam*A') * (I-gam^(len_epi+1-l)*A'^(len_epi+1-l))
        # A^l p
        Al_p = A^l * p
        # diag(p + A p + ... + A^l p)
        diag += Al_p
        # [diag(p) [I + ... + gam^L A.T^L] + ... + diag(A^L p) [I]]
        block += Diagonal(Al_p) * mat_sum
    end

    if return_diag
        return (block*r-diag.*q), diag
    else
        return (block*r-diag.*q)
    end
end


function contraction(q_opt, q_old, q_new)
    """
    explain
    """

    # neglect q-values that were not updated
    new_els = findall(!iszero, q_new-q_old)
    q = q_opt[new_els]
    q_o = q_old[new_els]
    q_n = q_new[new_els]

    return map(abs, q.-q_n) ./ map(abs, q.-q_o)
end