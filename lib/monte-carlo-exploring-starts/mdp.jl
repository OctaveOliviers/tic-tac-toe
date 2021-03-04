# @Created by: OctaveOliviers
# @Created on: 2021-01-15 18:50:08
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-03-04 18:29:14


# import libraries
using Random
using LinearAlgebra

# include files
include("utils.jl")


struct MDP
    """
    struct for Markov Decision Process
    """
    
    num_s       ::Int64
    num_sa      ::Int64
    discount    ::Float64
    structure   ::Matrix{Int8}
    policy      ::Matrix{Int8}
    transitions ::Matrix{Float64}
    rewards     ::Vector{Float64}
    q           ::Vector{Float64}
    terminal_state_action ::Vector{Int64}
    
    function MDP(num_s, num_sa, discount, structure, policy, transitions, rewards, q, terminal_state_action)
        """
        constructor
        """
        # assert that all the variables have the appropriate size
        @assert(all(map(!iszero, [num_s, num_sa, discount])))
        @assert(size(structure) == size(policy) == size(transitions') == (num_sa, num_s))
        @assert(size(rewards) == size(q) == (num_sa,))

        new(num_s, num_sa, discount, structure, policy, transitions, rewards, q, terminal_state_action)
    end
end # struct MDP


function generate_mdp(num_sa, num_s; discount=0.9, seed=42)
    """
    explain
    """
    # set random number generator
    Random.seed!(seed)

    # number of terminal states
    num_term = rand(1:ceil(num_s/3))

    # create structure
    structure = create_structure(num_s, num_sa, num_term)

    # create transition probabilities
    transitions = create_transitions(num_s, num_sa)

    # store terminal states of MDP
    term_states = [i for i=(num_s-num_term+1):num_s]

    # set q-values
    q = initialize_q(num_s, num_sa, num_term)
    
    # optimal policy
    policy = compute_policy(structure, q)

    # compute rewards
    rewards = (I - discount*transitions'*policy') * q

    mdp = MDP(num_s, num_sa, discount, structure, policy, transitions, rewards, q, term_states)

    return mdp
end


function create_structure(num_s, num_sa, num_term)
    """
    explain
    """
    # assert that there are not too many (state, action) pairs
    @assert num_s <= num_sa <= num_s^2

    # structure matrix
    structure = zeros(Int8, num_sa, num_s)
    # ensure each state has at least one action
    for s = 1:num_s ; structure[s,s] = 1 ; end
    # assign the other actions randomly
    for sa = (num_s+1):num_sa ; structure[sa, Int(rand(1:(num_s-num_term)))] = 1 ; end

    return structure
end


function create_transitions(num_s, num_sa)
    """
    explain
    """
    # transition matrix
    transitions = [I rand(Float64, num_s, num_sa-num_s)]
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


function initialize_q(num_s, num_sa, num_term)
    """
    explain
    """
    # q-values
    q = rand(Float64, num_sa)
    # q-values in terminal states are zero
    q[Int(num_s-num_term+1):num_s] .= 0

    return q
end


function compute_policy(structure, q)
    """
    explain
    """
    # extract useful info
    num_sa, num_s = size(structure)

    # policy matrix
    policy = zeros(Int64, num_sa, num_s)
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


function policy_q(policy, transitions, rewards, discount)
    """
    explain
    """
    return (I-discount*transitions'*policy')\rewards
end


function generate_episode(mces, mdp; max_len_epi = 5*1e2)
    """
    explain
    """
    # store the state-actions and rewards encountered in the episode
    sa = Array{Int64}(undef, 0)
    r = Array{Float64}(undef, 0)
    # store number of visits of each state-action in this episode
    n_vis = zeros(Int64, mdp.num_sa)

    # choose initial state-action pair according to weights in prior
    append!(sa, sample([i for i = 1:mdp.num_sa], Weights(mces.prior)))
    # store reward of initial state-action
    append!(r, mdp.rewards[sa[end]])
    # update number of visits of initial state-action
    n_vis[sa[end]] += 1

    # generate an episode from initial state
    while !(sa[end] in mdp.terminal_state_action) && (length(sa) <= max_len_epi)
        # go to new state-action
        append!(sa, sample([i for i = 1:mdp.num_sa], Weights((mces.policy*mdp.transitions)[:,sa[end]])))
        # store reward of new state-action
        append!(r, mdp.rewards[sa[end]])
        # update number of visits of new state-action
        n_vis[sa[end]] += 1
    end

    return sa, r, n_vis
end