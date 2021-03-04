# @Created by: OctaveOliviers
# @Created on: 2021-01-15 17:36:10
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-03-04 20:12:23


# import libraries
using Random
using StatsBase
using ProgressBars

# include files
include("utils.jl")


mutable struct MCES
    """
    struct for Monte Carlo Exploring Starts solver
    """
    
    q       ::Vector{Float64}
    policy  ::Matrix{Int8}
    prior   ::Vector{Float64}
    
    function MCES(q, policy, prior)
        """
        constructor
        """
        # assert that all the variables have the appropriate size
        @assert(size(q) == size(prior))
        @assert(size(q) == (size(policy)[1],))

        new(q, policy, prior)
    end
end # struct MCES


function run_mces!(mces, mdp; num_epi = 1e3, max_len_epi = 2*1e2, seed = 42, tol = 1e-2, c = Array{Float64}(undef, 0))
    """
    explain
    """
    # set random number generator
    Random.seed!(seed)

    # # initialize vector of q-values
    # mces.q = 100*rand(Float64, mdp.num_sa)
    # # compute policy matrix from q-values
    # mces.policy = compute_policy(mdp.structure, mces.q)
    # number of times that each state-action is visited
    tot_vis = zeros(Int64, mdp.num_sa)

    # loop until convergence
    for k = ProgressBar(1:num_epi)
        # run one simulation down the MDP and update the parameters
        mces_step!(mces, mdp, tot_vis, max_len_epi=max_len_epi)

        # check for convergence
        if all(abs.(mdp.q-mces.q) .< tol)
            break
        end

        # compute contraction
        append!(c, maximum(abs.(mdp.q-mces.q)))
    end
end


function mces_step!(mces, mdp, tot_vis; max_len_epi=2*1e2)
    """
    explain
    """
    # generate episode
    epi_sa, epi_r, epi_vis = generate_episode(mces, mdp, max_len_epi=max_len_epi)
    # update q-estimates
    update_q_values!(mces.q, epi_sa, epi_r, tot_vis, epi_vis, mdp.discount)
    # compute policy matrix from q-values
    update_policy!(mces.policy, mdp.structure, mces.q)
end


function update_q_values!(q, episode_sa, episode_r, total_visits, episode_visits, discount)
    """
    explain
    """
    # update total number of visits of each state-action
    total_visits .+= episode_visits

    # goal value
    g = 0.
    # update backwards
    reverse!(episode_sa)
    reverse!(episode_r)
    # loop over each step of the episode
    for t in 1:length(episode_sa)
        # update goal value
        g = discount*g + episode_r[t]

        # update q-estimate with incremental mean formula
        episode_visits[episode_sa[t]] -= 1
        step_size = 1. / (total_visits[episode_sa[t]] - episode_visits[episode_sa[t]])
        
        q[episode_sa[t]] += step_size*(g - q[episode_sa[t]])
    end
end


function update_q_values(q, episode_sa, episode_r, total_visits, episode_visits, discount)
    """
    explain
    """
    # new q-values
    q_new = deepcopy(q)
    # update q-values in place
    update_q_values!(q_new, episode_sa, episode_r, total_visits, episode_visits, discount)
    
    return q_new
end


function update_policy!(policy, structure, q)
    """
    explain
    """
    policy .= compute_policy(structure, q)
end


function update_policy(policy, structure, q)
    """
    explain
    """
    # new policy
    policy_new = deepcopy(policy)
    # update policy in place
    update_policy!(policy_new, structure, q)

    return policy_new
end


function expected_mces(mces, mdp; num_epi=1e3, max_len_epi=2*1e2, seed=42)
    """
    explain
    """
    # set random number generator
    Random.seed!(seed)

    # extract useful info
    num_sa,num_s = size(structure)

    # initialize vector of q-values
    q = 100*rand(Float64, num_sa)
    # compute policy matrix from q-values
    P = compute_policy(structure, q)
    # diagonal with all visits
    diag_sum = zeros(Float64, num_sa)
    diag = zeros(Float64, num_sa)

    # # if debugging
    # q_old = deepcopy(q)
    # P_old = deepcopy(P)
    # better = true
    # P_all = deepcopy(P)

    # loop until convergence
    for k = ProgressBar(1:num_epi)
    # for k = 1:num_epi

        # compute step and number of visits
        stp, diag =  step(P, transitions, rewards, prior, q, discount, max_len_epi, return_diag=true)

        diag_sum += diag

        # update q-estimates
        q += stp./diag_sum
        # compute policy matrix from q-values
        P = compute_policy(structure, q)

        # pot, val, q_new = compute_potential(P, transitions, rewards, prior, q, discount, max_len_epi, return_q_opt=true)

        # if P != P_old

        #     v1 = P_old'*(q - stp./diag_sum)
        #     v2 = P_old'*q

        #     if k > 1
        #         # better *= prod( P'*q_new >= P_old'*q_old )
        #         better *= prod( v2 .>= v1 )
        #     end

        #     # a_new = (P-P_old+1)/2
        #     P_all[(P-P_old).==1] .+= 1

        #     @info "new policy" P
        #     println("v1 = ", v1)
        #     println("v2 = ", v2)
        #     # println("old values ", P_old'*q_old)
        #     # println("new values ", P'*q_new)
        #     @info better

        #     # # check order of the weights vs q_values
        #     # # p_w_qk = sortperm(1 .- diag./diag_sum)
        #     # # p_qk = sortperm(q_old)
        #     # # println("order weights qk, qk", [p_w_qk, p_qk])
        #     # println("\nprevious qk-values ")
        #     # show(stdout, "text/plain", structure .* (q - Diagonal(1 ./diag_sum)*stp))
        #     # println("\nnew q*k-values ")
        #     # show(stdout, "text/plain", structure .* q_new)
        #     # println("\nweights of qk ")
        #     # show(stdout, "text/plain", structure .* (1 .- diag./diag_sum))
        #     # println("\nweights of q*k ")
        #     # show(stdout, "text/plain", structure .* (diag./diag_sum))
 
        #     # println("\nPk q*k ", P_old'*q_old)
        #     # println("Pk+1 q*k ", P'*q_old)

        #     # println("\n----------------------------------------\n")
        # end

        # q_old = q_new
        # P_old = P

        # # if P != P_old
        # #     @info "new policy"
        # # end
        # # P_old = P

        # # pot, val = compute_potential(P, transitions, rewards, prior, q, discount, max_len_epi)
        # # println("potential is $pot")
        # # println("optimal values are $(round.(val, digits=3))")
        # # println("current values are $(round.(P'*q, digits=3))")
    end

    show(stdout, "text/plain", diag./diag_sum)    

    # println("Was the new policy always better? ", better)

    # println("\nsum of all experienced policies")
    # show(stdout, "text/plain", P_all)

    return q

end