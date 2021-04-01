# @Created by: OctaveOliviers
# @Created on: 2021-01-15 17:36:10
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-04-01 14:53:56


# import libraries
using Random
using StatsBase
using ProgressBars

include("params.jl")


mutable struct MCES
    """
    struct for Monte Carlo Exploring Starts solver
    """
    
    q       ::Vector{Float64}
    policy  ::Matrix{Int8}
    prior   ::Vector{Float64}
    
    function MCES(
        mdp; 
        seed=nothing)
        """
        constructor
        """
        # set random number generator
        if seed !== nothing; Random.seed!(seed); end

        # initialise q-values
        q = rand(mdp.num_sa)
        # q = [5, 5.1, 0, 0]
        # initialise policy
        policy = compute_policy(mdp.structure, q)
        # initialise prior
        prior = normalize(rand(mdp.num_sa))

        new(q, policy, prior)
    end
end # struct MCES


function run_mces!(
    mces, 
    mdp; 
    num_epi=1e3, 
    max_len_epi=2*1e2, 
    seed=nothing, 
    tol=TOLERANCE, 
    c=Array{Float64}(undef, 0))
    """
    explain
    """
    # set random number generator
    if seed !== nothing; Random.seed!(seed); end

    # number of times that each state-action is visited
    tot_vis = zeros(Int64, mdp.num_sa)

    # loop until convergence
    for k = ProgressBar(1:num_epi)
        # run one simulation down the MDP and update the parameters
        mces_step!(mces, mdp, tot_vis, max_len_epi=max_len_epi, seed=rand(UInt64))

        # check for convergence
        if all(abs.(mdp.q-mces.q) .< tol)
            break
        end

        # compute contraction
        append!(c, maximum(abs.(mdp.q-mces.q)))
    end
end


function mces_step_all_sa_update!(
    mces, 
    mdp; 
    max_num_update=1e3, 
    max_len_epi=2*1e2, 
    seed=nothing)
    """
    explain
    """
    # assert exploring starts assumption
    @assert all(mces.prior .> 0)

    # set random number generator
    if seed !== nothing; Random.seed!(seed); end

    # number of times that each state-action is visited
    tot_vis = zeros(Int64, mdp.num_sa)

    # apply MCES update until every state-action has been visited once
    i = 1
    while any(tot_vis .< 1)
        # run one simulation down the MDP and update the parameters
        mces_step!(mces, mdp, tot_vis, max_len_epi=max_len_epi, seed=rand(UInt64))

        println("finished update ", i)

        if i >= max_num_update
            println("Terminate because reached maximal number of allowed updates.")
        end

        i += 1
    end    
end


function mces_step!(
    mces, 
    mdp, 
    tot_vis; 
    max_len_epi=2*1e2, 
    seed=nothing)
    """
    explain
    """
    # generate episode
    epi_sa, epi_r, epi_vis = generate_episode(mces, mdp, max_len_epi=max_len_epi, seed=seed)
    # update q-estimates
    update_q_values!(mces.q, epi_sa, epi_r, tot_vis, epi_vis, mdp.discount)
    # update policy matrix from q-values
    update_policy!(mces.policy, mdp.structure, mces.q)
end


function update_q_values!(
    q, 
    episode_sa, 
    episode_r, 
    total_visits,
    episode_visits, 
    discount)
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


function update_q_values(
    q, 
    episode_sa, 
    episode_r, 
    total_visits, 
    episode_visits, 
    discount)
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


function run_mces_exp!(mces, mdp; num_epi=1e3, max_len_epi=1e3, tol=TOLERANCE)
    """
    explain
    """
    # number of times that each state-action is visited
    tot_vis = zeros(Int64, mdp.num_sa)

    # loop until convergence
    for k = ProgressBar(1:num_epi)
        # perform one mces expected step
        mces_exp_step!(mces, mdp, tot_vis, max_len_epi=max_len_epi)

        # check for convergence
        if all(abs.(mdp.q-mces.q) .< tol)
            break
        end
    end
end


function mces_exp_step!(mces, mdp, tot_vis; max_len_epi=1e3)
    """
    explain
    """
    # episode step and number of visits
    epi_stp = compute_q_policy(mces.policy, mdp.transitions, mdp.rewards, mdp.discount) - mces.q
    epi_vis = sum([(mces.policy*mdp.transitions)^i*mces.prior for i = 0:max_len_epi])

    tot_vis += epi_vis

    # update q-estimates
    mces.q += epi_stp.*epi_vis./tot_vis
    # compute policy matrix from q-values
    mces.policy = compute_policy(mdp.structure, mces.q)
end