# @Created by: OctaveOliviers
# @Created on: 2021-03-04 17:36:54
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-03-04 18:43:20

# test convergence of Example 5.12 in pp. 234-236 of Bertsekas & Tsitsiklis (1996) 

using Dates
using Random

Random.seed!(Dates.day(today()))

include("mdp.jl")
include("mces.jl")
include("utils.jl")

gam = 0.95

S = [[1 0];
     [1 0];
     [0 1];
     [0 1]]

T = [[1 0 0 1];
     [0 1 1 0]]

r = [0., -1., 0., -1.]

P = [[1 0];
     [0 0];
     [0 1];
     [0 0]]

q = policy_q(P, T, r, gam)

mdp = MDP(2, 4, gam, S, P, T, r, q, [])


q0 = 100*rand(Float64, 4)
P0 = compute_policy(S, q0)
# P = [[1 0];
#      [0 0];
#      [0 1];
#      [0 0]]
prior = normalize(rand(Float32, mdp.num_sa), 1)
mces = MCES(q0, P0, prior)

c = Array{Float64}(undef, 0)
run_mces!(mces, mdp, num_epi=1e7, max_len_epi=1e3, seed=10, c=c)

@info "results" mces.q mdp.q