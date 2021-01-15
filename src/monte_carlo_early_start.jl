# @Author: OctaveOliviers
# @Date:   2021-01-15 17:36:10
# @Last Modified by:   OctaveOliviers
# @Last Modified time: 2021-01-15 19:25:14

# import libraries
using Random
using StatsBase
using Dates
using ProgressBars

function monte_carlo_early_start(structure, transitions, rewards, prior, discount; num_epi = 10000, len_epi = 20)
    """
    explain
    """

    # set random number generator
    Random.seed!(Dates.day(today()))

    # extract useful info
    num_sa,num_s = size(structure)

    # initialize vector of q-values
    q = 100*rand(Float64, num_sa)
    # compute policy matrix from q-values
    P = compute_policy(structure, q)
    # number of times that each state-action is visited
    n = zeros(Int64, num_sa)

    # loop until convergence
    for k = ProgressBar(1:num_epi)
        # store the state-actions and rewards encountered in the episode
        z = Array{Int64}(undef, 0)
        r = Array{Int64}(undef, 0)
        # store number of visits of each state-action in this episode
        n_epi = zeros(Int64, num_sa)

        # choose initial state-action pair according to weights in prior
        append!(z, sample([i for i = 1:num_sa], Weights(prior)))
        # store reward of initial state-action
        append!(r, rewards[z[end]])
        # update number of visits of initial state-action
        n_epi[z[end]] += 1

        # generate an episode from initial state
        for t in 1:len_epi
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
    end

    return q
end


function compute_policy(S, q)
    """
    explain
    """

    # extract useful info
    num_sa,num_s = size(S)

    # policy matrix
    P = zeros(Int64, num_sa, num_s)
    # for each state choose action with highest q-value
    for s = 1:num_s
        # find actions of that state
        sa = findall(!iszero, S[:,s])
        # index of maximal q value in state s
        idx_max_q = argmax(q[sa])
        # choose the action with maximal q-value
        P[sa[idx_max_q], s] = 1
    end

    return P
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