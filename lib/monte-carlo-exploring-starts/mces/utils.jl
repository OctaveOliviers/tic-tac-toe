# import libraries
using Random
using StatsBase
using ProgressBars

include("mces.jl")
include("../mdp/mdp.jl")
include("../params.jl")


function run_mces!(
    mces::MCES,
    mdp::MDP;
    max_num_epi::Real=EPISODE_MAX_NUMBER,
    max_len_epi::Real=EPISODE_MAX_LENGTH,
    tol::Real=TOLERANCE,
    seed::Integer=NO_SEED
    )::Nothing
    # c=Array{Float64}(undef, 0))
    """
    explain
    """
    # set random number generator
    if seed != NO_SEED; Random.seed!(seed); end

    # number of times that each state-action is visited
    tot_vis = zeros(Integer, mdp.num_sa)

    # loop until convergence
    for k = ProgressBar(1:max_num_epi)
        # run one simulation down the MDP and update the parameters
        mces_step!(mces, mdp, tot_vis, max_len_epi=max_len_epi, seed=rand(UInt64))

        # check for convergence
        if all(abs.(mdp.q-mces.q) .< tol); break; end

        # compute contraction
        # append!(c, maximum(abs.(mdp.q-mces.q)))
    end

    return nothing
end


function mces_step_all_sa_update!(
    mces::MCES,
    mdp::MDP;
    max_num_update::Real=EPISODE_MAX_NUMBER,
    max_len_epi::Real=EPISODE_MAX_LENGTH,
    seed::Integer=NO_SEED
    )::Nothing
    """
    explain
    """
    # assert exploring starts assumption
    @assert all(mces.prior .> 0)

    # set random number generator
    if seed != NO_SEED; Random.seed!(seed); end

    # number of times that each state-action is visited
    tot_vis = zeros(Integer, mdp.num_sa)

    # apply MCES update until every state-action has been visited once
    while any(tot_vis .< 1)
        # run one simulation down the MDP and update the parameters
        mces_step!(mces, mdp, tot_vis, max_len_epi=max_len_epi, seed=rand(UInt64))

        if sum(tot_vis) >= max_num_update*max_len_epi;
            println("reached max number of updates.")
            break
        end
    end

    return nothing
end


function mces_step!(
    mces::MCES,
    mdp::MDP,
    total_visits::Vector;
    max_len_epi::Real=EPISODE_MAX_LENGTH,
    seed::Integer=NO_SEED
    )::Nothing
    """
    explain
    """
    # generate episode
    epi_sa, epi_r, epi_vis = generate_episode(mces, mdp, max_len_epi=max_len_epi, seed=seed)
    # update q-estimates
    update_q_values!(mces.q, epi_sa, epi_r, total_visits, epi_vis, mdp.discount)
    # update policy matrix from q-values
    update_policy!(mces.policy, mdp.structure, mces.q)

    return nothing
end


function update_q_values!(
    q::Vector,
    episode_sa::Vector,
    episode_r::Vector,
    total_visits::Vector,
    episode_visits::Vector,
    discount::Real
    )::Nothing
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

    return nothing
end


function update_q_values(
    q::Vector,
    episode_sa::Vector,
    episode_r::Vector,
    total_visits::Vector,
    episode_visits::Vector,
    discount::Real
    )::Vector
    """
    explain
    """
    # new q-values
    q_new = deepcopy(q)
    # update q-values in place
    update_q_values!(q_new, episode_sa, episode_r, total_visits, episode_visits, discount)

    return q_new
end


function update_policy!(
    policy::Matrix,
    structure::Matrix,
    q::Vector
    )::Nothing
    """
    explain
    """
    policy .= compute_policy(structure, q)

    return nothing
end


function update_policy(
    policy::Matrix,
    structure::Matrix,
    q::Vector
    )::Matrix
    """
    explain
    """
    # new policy
    policy_new = deepcopy(policy)
    # update policy in place
    update_policy!(policy_new, structure, q)

    return policy_new
end


function run_mces_exp!(
    mces::MCES,
    mdp::MDP;
    max_num_epi::Real=EPISODE_MAX_NUMBER,
    max_len_epi::Real=EPISODE_MAX_LENGTH,
    tol::Real=TOLERANCE
    )::Nothing
    """
    explain
    """
    # number of times that each state-action is visited
    tot_vis = zeros(Integer, mdp.num_sa)

    # loop until convergence
    for k = ProgressBar(1:max_num_epi)
        # perform one mces expected step
        mces_exp_step!(mces, mdp, tot_vis, max_len_epi=max_len_epi)

        # check for convergence
        if all(abs.(mdp.q-mces.q) .< tol); break; end
    end

    return nothing
end


function mces_exp_step!(
    mces::MCES,
    mdp::MDP,
    total_visits::Vector;
    max_len_epi::Real=EPISODE_MAX_LENGTH
    )::Nothing
    """
    explain
    """
    # episode step and number of visits
    epi_stp = compute_q_policy(mces.policy, mdp.transitions, mdp.rewards, mdp.discount) - mces.q
    epi_vis = sum([(mces.policy*mdp.transitions)^i*mces.prior for i = 0:max_len_epi])

    total_visits += epi_vis

    # update q-estimates
    mces.q += epi_stp.*epi_vis./total_visits
    # compute policy matrix from q-values
    mces.policy = compute_policy(mdp.structure, mces.q)

    return nothing
end
