# @Created by: OctaveOliviers
# @Created on: 2021-01-15 17:36:10
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-02-02 14:14:53


# import libraries
using Random
using StatsBase
using Dates
using ProgressBars

# include files
include("utils.jl")


function monte_carlo_exploring_start(structure, transitions, rewards, prior, discount; num_epi = 2e2, max_len_epi = 2e2)
    """
    explain
    """

    # set random number generator
    @debug "Set seed" Random.seed!(Dates.day(today()))

    # extract useful info
    num_sa,num_s = size(structure)
    # terminal states
    term_s = findall(iszero, dropdims(sum(structure*transitions-I, dims=1), dims=1))

    # initialize vector of q-values
    q = 100*rand(Float64, num_sa)
    # q = [10., 3., 5., -4., -10., 0., 0., 0.] ;
    # compute policy matrix from q-values
    P = compute_policy(structure, q)
    # number of times that each state-action is visited
    n = zeros(Int64, num_sa)

    # loop until convergence
    for k = ProgressBar(1:num_epi)
    # for k = 1:num_epi

        # store the state-actions and rewards encountered in the episode
        z = Array{Int64}(undef, 0)
        r = Array{Float64}(undef, 0)
        # store number of visits of each state-action in this episode
        n_epi = zeros(Int64, num_sa)

        # choose initial state-action pair according to weights in prior
        append!(z, sample([i for i = 1:num_sa], Weights(prior)))
        # store reward of initial state-action
        append!(r, rewards[z[end]])
        # update number of visits of initial state-action
        n_epi[z[end]] += 1

        # generate an episode from initial state
        # for t in 1:len_epi
        while ~ (z[end] in term_s) && (length(z) <= max_len_epi)
            # go to new state-action
            append!(z, sample([i for i = 1:num_sa], Weights((P*transitions)[:,z[end]])))
            # store reward of new state-action
            append!(r, rewards[z[end]])
            # update number of visits of new state-action
            n_epi[z[end]] += 1
        end

        # update total number of visits of each state-action
        n += n_epi

        # update q-estimates
        q = update_q_values(q, z, r, n, n_epi, discount)
        # compute policy matrix from q-values
        P = compute_policy(structure, q)


        # @info z r q

        # pot, val = compute_potential(P, transitions, rewards, prior, q, discount, max_len_epi)
        # println("potential is $pot")
        # print("value is $val")
    end

    return q
end

function expected_mces(structure, transitions, rewards, prior, discount; num_epi = 1e3, max_len_epi = 4*1e2)
    """
    explain
    """

    # set random number generator
    @debug "Set seed" Random.seed!(Dates.day(today()))

    # extract useful info
    num_sa,num_s = size(structure)

    # initialize vector of q-values
    q = 100*rand(Float64, num_sa)
    # compute policy matrix from q-values
    P = compute_policy(structure, q)
    # diagonal with all visits
    diag_sum = n = zeros(Float64, num_sa)

    # if debugging
    q_old = deepcopy(q)
    P_old = deepcopy(P)
    better = true
    P_all = deepcopy(P)

    # loop until convergence
    # for k = ProgressBar(1:num_epi)
    for k = 1:num_epi

        # compute step and number of visits
        stp, diag =  step(P, transitions, rewards, prior, q, discount, max_len_epi, return_diag=true)

        diag_sum += diag

        # update q-estimates
        q += stp./diag_sum
        # compute policy matrix from q-values
        P = compute_policy(structure, q)

        pot, val, q_new = compute_potential(P, transitions, rewards, prior, q, discount, max_len_epi, return_q_opt=true)

        if P != P_old

            v1 = P_old'*(q - stp./diag_sum)
            v2 = P_old'*q

            if k > 1
                # better *= prod( P'*q_new >= P_old'*q_old )
                better *= prod( v2 .>= v1 )
            end

            # a_new = (P-P_old+1)/2
            P_all[(P-P_old).==1] .+= 1

            @info "new policy" P
            println("v1 = ", v1)
            println("v2 = ", v2)
            # println("old values ", P_old'*q_old)
            # println("new values ", P'*q_new)
            @info better

            # # check order of the weights vs q_values
            # # p_w_qk = sortperm(1 .- diag./diag_sum)
            # # p_qk = sortperm(q_old)
            # # println("order weights qk, qk", [p_w_qk, p_qk])
            # println("\nprevious qk-values ")
            # show(stdout, "text/plain", structure .* (q - Diagonal(1 ./diag_sum)*stp))
            # println("\nnew q*k-values ")
            # show(stdout, "text/plain", structure .* q_new)
            # println("\nweights of qk ")
            # show(stdout, "text/plain", structure .* (1 .- diag./diag_sum))
            # println("\nweights of q*k ")
            # show(stdout, "text/plain", structure .* (diag./diag_sum))
 
            # println("\nPk q*k ", P_old'*q_old)
            # println("Pk+1 q*k ", P'*q_old)

            # println("\n----------------------------------------\n")
        end

        q_old = q_new
        P_old = P

        # if P != P_old
        #     @info "new policy"
        # end
        # P_old = P

        # pot, val = compute_potential(P, transitions, rewards, prior, q, discount, max_len_epi)
        # println("potential is $pot")
        # println("optimal values are $(round.(val, digits=3))")
        # println("current values are $(round.(P'*q, digits=3))")
    end

    println("Was the new policy always better? ", better)

    println("\nsum of all experienced policies")
    show(stdout, "text/plain", P_all)

    return q

end


function update_q_values(q, z, r, n, n_epi, gam)
    """
    explain
    """

    # extract useful info
    len_epi = length(z)

    # goal value
    g = 0
    # update backwards
    reverse!(z)
    reverse!(r)
    # loop over each step of the episode
    for t in 1:len_epi
        # update goal value
        g = gam*g + r[t]

        # update q-estimate with incremetnal mean formula
        n_epi[z[t]] -= 1
        q[z[t]] += (g - q[z[t]]) / (n[z[t]] - n_epi[z[t]])
    end
    
    return q
end