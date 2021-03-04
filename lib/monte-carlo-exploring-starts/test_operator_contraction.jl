# @Created by: OctaveOliviers
# @Created on: 2021-03-01 13:03:44
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-03-04 17:06:38

using Dates

include("mces.jl")
include("mdp.jl")
include("utils.jl")

Random.seed!(Dates.day(today()))

# build Markov Decision Process
base = 5
num_s = base + convert(Int64, ceil(base*rand()))
num_sa = convert(Int64, num_s + ceil((num_s-1)*num_s*rand()))
# mdp
mdp = generate_mdp(num_sa, num_s, discount=0.9*rand(), seed=Dates.day(today()))

@info "Parameters are" mdp.num_s mdp.num_sa mdp.discount


# build Monte Carlo Exploring Starts solver
# prior
prior = normalize(rand(Float32, mdp.num_sa), 1)

# generate two random q-vectors
q1 = rand(Float64, num_sa)
P1 = compute_policy(mdp.structure, q1)
q1_opt = policy_q(P1, mdp.transitions, mdp.rewards, mdp.discount)
# w1 = rand(Float64, num_sa)
# w1 = w1/sum(w1)
# 
q2 = rand(Float64, mdp.num_sa)
P2 = compute_policy(mdp.structure, q2)
q2_opt = policy_q(P2, mdp.transitions, mdp.rewards, mdp.discount)
# w2 = rand(Float64, num_sa)
# w2 = w2/sum(w2)
# update weights
# w = rand(Float64, num_sa)
# w = w/sum(w)

old_diff = maximum(abs.(q1-q2))

println("Different policies? ", P1 != P2)

# apply mces operator
# q1_new = (1 .- w).*q1 + w.*q1_opt 
# q2_new = (1 .- w).*q2 + w.*q2_opt

# ụdpate q1 until each state action has been updated at least once
tot_vis = zeros(Int64, num_sa)
while any(iszero, tot_vis)
    epi_sa, epi_r, epi_vis = generate_episode(P1, transitions, rewards, prior, term_states; max_len_epi = 5*1e2)
    update_q_values!(q1, epi_sa, epi_r, tot_vis, epi_vis, discount)
    P1 = compute_policy(structure, q1)
end

# ụdpate q2
tot_vis = zeros(Int64, num_sa)
while any(iszero, tot_vis)
    epi_sa, epi_r, epi_vis = generate_episode(P2, transitions, rewards, prior, term_states; max_len_epi = 5*1e2)
    update_q_values!(q2, epi_sa, epi_r, tot_vis, epi_vis, discount)
    P2 = compute_policy(structure, q2)
end

new_diff = maximum(q1-q2)

println("max|q1-q2| >= max|Bq1-Bq2|? ", old_diff >= new_diff)
println("old ", old_diff)
println("new ", new_diff)

# conclusion
#   this operator does not always contract when w are random