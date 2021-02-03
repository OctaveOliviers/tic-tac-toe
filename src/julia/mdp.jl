# @Created by: OctaveOliviers
# @Created on: 2021-01-15 18:50:08
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-02-02 14:14:55


# import libraries
using Random
using LinearAlgebra
using Dates

# include files
include("utils.jl")


function gen_mdp(num_sa, num_s; gam=0.9)
    """
    explain
    """
    
    # set random number generator
    @debug "Set seed" Random.seed!(Dates.day(today()))

    # number of terminal states
    num_ter = rand(1:ceil(num_s/3))

    # create structure
    S = zeros(Int8, num_sa, num_s)
    # ensure each state has at least one action
    for s = 1:num_s ; S[s,s] = 1 ; end
    # assign the other actions randomly
    for sa = (num_s+1):num_sa ; S[sa, Int(rand(1:(num_s-num_ter)))] = 1 ; end

    # set q-values
    q = rand(Float64, num_sa)
    # q-values in terminal states are zero
    q[Int(num_s-num_ter+1):num_s] .= 0

    # optimal policy
    P = compute_policy(S, q)

    # create transition probabilities
    T = [I rand(Float64, num_s, num_sa-num_s)]
    # normalize
    T = T ./ sum(T, dims=1)

    # compute rewards
    r = (I - gam*T'*P') * q

    return S, T, P, r, q, gam
end