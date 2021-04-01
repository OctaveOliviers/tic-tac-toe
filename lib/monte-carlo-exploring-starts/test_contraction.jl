# @Created by: OctaveOliviers
# @Created on: 2021-04-01 11:27:07
# @Last Modified by: OctaveOliviers
# @Last Modified on: 2021-04-01 14:54:01

include("mdp.jl")
include("mces.jl")

SEED = 42

# create markov ecision problem
num_s, num_sa = 5, 15
mdp = generate_mdp(num_sa, num_s; discount=0.9, seed=42)

# create two MCES solvers
mces1 = MCES(mdp, seed=13)
mces2 = MCES(mdp, seed=11)

# apply MCES steps on each
mces_step_all_sa_update!(mces1, mdp; seed=nothing)
mces_step_all_sa_update!(mces2, mdp; seed=nothing)

# compute base distance
old_dist = maximum(abs.(mces1.q - mces2.q))
println(old_dist)

# update
mces_step_all_sa_update!(mces1, mdp; seed=nothing)
mces_step_all_sa_update!(mces2, mdp; seed=nothing)

# compute new distance
new_dist = maximum(abs.(mces1.q - mces2.q))
println(new_dist)