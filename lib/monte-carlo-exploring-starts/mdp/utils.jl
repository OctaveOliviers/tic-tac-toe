

# import libraries
using Random
using LinearAlgebra

include("mdp.jl")
include("../mces/mces.jl")
include("../params.jl")


function generate_mdp(
    num_sa::Integer,
    num_s::Integer;
    discount::Real=DISCOUNT,
    seed::Integer=NO_SEED
    )::MDP
    """
    explain
    """
    # set random number generator
    if seed != NO_SEED; Random.seed!(seed); end

    # number of terminal states
    num_term = rand(1:Int(ceil(num_s/3)))
    # create structure
    structure = create_structure(num_s, num_sa, num_term)
    # create transition probabilities
    transitions = create_transitions(num_s, num_sa, num_term)
    # store terminal states of MDP
    term_states = [i for i=(num_s-num_term+1):num_s]
    # set q-values
    q = initialize_q(num_s, num_sa, num_term)
    # optimal policy
    policy = compute_policy(structure, q)
    # compute rewards
    rewards = (I - discount*transitions'*policy') * q

    return MDP(num_s, num_sa, discount, structure, policy, transitions, rewards, q, term_states)
end


function create_structure(
    num_s::Integer,
    num_sa::Integer,
    num_term::Integer
    )::Matrix
    """
    explain
    """
    # assert that there are not too many (state, action) pairs
    @assert num_s <= num_sa <= num_s^2

    # structure matrix
    structure = zeros(Integer, num_sa, num_s)
    # ensure each state has at least one action
    for s = 1:num_s ; structure[s,s] = 1 ; end
    # assign the other actions randomly
    for sa = (num_s+1):num_sa ; structure[sa, Int(rand(1:(num_s-num_term)))] = 1 ; end

    return structure
end


function create_transitions(
    num_s::Integer,
    num_sa::Integer,
    num_term::Integer
    )::Matrix
    """
    explain
    """
    # transition matrix
    # need first identity to ensure that a terminal state has zero reward
    transitions = [I rand(num_s, num_sa-num_s)]
    transitions[:,1:num_s-num_term] = rand(num_s, num_s-num_term)
    # normalize
    transitions = transitions ./ sum(transitions, dims=1)

    return transitions
end


# function terminal_states(structure, transitions)
#     """
#     explain
#     """
#     return findall(iszero, dropdims(sum(structure*transitions-I, dims=1), dims=1))
# end


function initialize_q(
    num_s::Integer,
    num_sa::Integer,
    num_term::Integer
    )::Vector
    """
    explain
    """
    # q-values
    q = 5*rand(Float64, num_sa)
    # q-values in terminal states are zero
    q[num_s-num_term+1:num_s] .= 0

    return q
end


function compute_policy(
    structure::Matrix,
    q::Vector
    )::Matrix
    """
    explain
    """
    # extract useful info
    num_sa, num_s = size(structure)

    # policy matrix
    policy = zeros(Int8, num_sa, num_s)
    # for each state choose action with highest q-value
    for s = 1:num_s
        # find actions of that state
        sa = findall(!iszero, structure[:,s])
        # index of maximal q value in state s
        idx_max_q = argmax(q[sa])
        # choose the action with maximal q-value
        policy[sa[idx_max_q], s] = 1
    end

    return policy
end


function compute_q_policy(
    policy::Matrix,
    transitions::Matrix,
    rewards::Vector,
    discount::Real
    )::Vector
    """
    explain
    """
    # solve Bellman equation
    return (I-discount*transitions'*policy')\rewards
end


function generate_episode(
    mces::MCES,
    mdp::MDP;
    max_len_epi::Real=EPISODE_MAX_LENGTH,
    seed::Integer=NO_SEED
    )::Tuple{Vector,Vector,Vector}
    """
    explain
    """
    # set random number generator
    if seed != NO_SEED; Random.seed!(seed); end

    # store the state-actions and rewards encountered in the episode
    sa, r = Vector{Integer}(undef, 0), Vector{AbstractFloat}(undef, 0)
    # store number of visits of each state-action in this episode
    n_vis = zeros(Integer, mdp.num_sa)

    function take_step(transition_prob)
        # choose state-action pair according to weights in transition_prob
        append!(sa, sample(1:mdp.num_sa, Weights(transition_prob)))
        # store reward of initial state-action
        append!(r, mdp.rewards[sa[end]])
        # update number of visits of initial state-action
        n_vis[sa[end]] += 1
    end

    # initialise episode
    take_step(mces.prior)
    # generate an episode from initial state
    while !(sa[end] in mdp.terminal_state_action)
        # go to new state-action
        take_step((mces.policy*mdp.transitions)[:,sa[end]])
        # exit if episode is too long
        if length(sa) >= max_len_epi
            println("Terminate because reached maximal length of episode.")
            break
        end
    end

    return sa, r, n_vis
end
