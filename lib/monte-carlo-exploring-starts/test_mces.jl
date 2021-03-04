# @Created by: OctaveOliviers
# @Created on: 2021-03-03 20:29:24
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-03-04 17:43:53

include("mdp.jl")
include("mces.jl")
include("utils.jl")

gam = 0.9 ;
p = [0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1]

S = [[1 0 0 0 0 0];
     [0 1 0 0 0 0];
     [0 1 0 0 0 0];
     [0 0 1 0 0 0];
     [0 0 1 0 0 0];
     [0 0 0 1 0 0];
     [0 0 0 0 1 0];
     [0 0 0 0 0 1]]

T = [[0 0 0 0 0 0 0 0];
     [1 0 0 .8 0 0 0 0];
     [0 0 1 0 0 0 0 0];
     [0 1 0 0 0 1 0 0];
     [0 0 0 0 1 0 1 0];
     [0 0 0 .2 0 0 0 1]]

r = [0., 10., 0., -10., -20., 0., 0., 0.] ;

P = [[1 0 0 0 0 0];
     [0 1 0 0 0 0];
     [0 0 0 0 0 0];
     [0 0 1 0 0 0];
     [0 0 0 0 0 0];
     [0 0 0 1 0 0];
     [0 0 0 0 1 0];
     [0 0 0 0 0 1]]
q = policy_q(P, T, r, gam)

mdp = MDP(6, 8, gam, S, P, T, r, q, [6,7,8])

q0 = 100*rand(Float64, 8)
P0 = compute_policy(S, q0)
mces = MCES(q0, P0, p)

run_mces!(mces, mdp, num_epi=1e6, seed=42)

@info "results" mces.q mdp.q