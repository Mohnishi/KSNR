"""
This file is for executing Gym Walker2d task with CEM.
Smooth dynamics.
"""

#------------------Packages-------------------#
using LinearAlgebra, Random, Statistics
BLAS.set_num_threads(32)
using Plots, JLSO
using LyceumBase.Tools, LyceumBase, LyceumAI, LyceumMuJoCo, MuJoCo, UniversalLogger, Shapes
using FastClosures, Distributions, Parameters
using UnsafeArrays
using Distances


include("../utils/weightmat.jl")
include("../mujoco_models/walker2d.jl")
include("../planner/PolicySelect-GT.jl")
include("../planner/PolicySelect-SP.jl")
#-----------------------Constants -------------------#
numfeatures = 200      # dimH  (for state x)
T           = 300      # task horizon (H in the note)
samplecem   = 300      # sample number of CEM every iteration
elitenum    = 20       # Elite number of CEM
cemiter     = 50        # number of iterations for CEM

@info "Walker2d Task"
@info "Seed number" seednum
Random.seed!(seednum)
#----------------------Environment-------------------#
mjenvs = tconstruct(Walker2d, Threads.nthreads());
env = Walker2d()
dobs, dact = length(obsspace(env)), length(actionspace(env))
dpolicyparam = 300

PolicyMat  = PredictionMat(dpolicyparam, dact)  # Policy is Theata pi(x) and this is pi
KoopmanMat = PredictionMat(numfeatures, numfeatures)  

@info "Dimensions of observations/actions/outputs" dobs dact #dstate
@info "Simulation timestep:" LyceumBase.timestep(env)

#----------------------Define Functions-------------------#
# sum of abs of eigvals  using svd to speedup
function lambdacost(Y::AbstractArray,X::AbstractArray,arg)
    svdx = svd(X)
    U = svdx.U[:,1:end]
    V = svdx.V[:,1:end]
    S = diagm(svdx.S[1:end])
    5 * sum(abs.(eigvals(U'*Y*V*S/(S.^2+I))))
end
# getting action using policy feature
function getaction!(
    action::AbstractVector,
    obs,
    featurize_pi,
    PolicyMatW::AbstractArray;
) 
    mul!(action, PolicyMatW, featurize_pi(obs))
    return action
end
#------------------Define Features-------------------#
include("../utils/rff.jl")
rffbandwidth_x = 5.0
rffbandwidth_pi = 30.0
const rff_x_lr = LinearRandom{Float64}(rffbandwidth_x, dobs, numfeatures)
const rff_pi_lr = LinearRandom{Float64}(rffbandwidth_pi, dobs, dpolicyparam)
#------------------Strategy Struct-------------------#
env_tconstructor = n -> tconstruct(Walker2d, n)
arg = 0

gt_polselect = PolicySelectGT(
        env_tconstructor = n -> tconstruct(Walker2d, n),
        phi_pol = rff_pi_lr,       
        samplecem = samplecem,   
        elitenum = elitenum,     
        cemiter = cemiter,       
        H = T,   #H
        featsize = dpolicyparam,   
        gamma = 1.,   #gamma  
    )

sp_polselect = PolicySelectSP(
        env_tconstructor = n -> tconstruct(Walker2d, n),
        phi_pol = rff_pi_lr, 
        phi_x  =  rff_x_lr,    
        arg = arg,     
        lambdacost = lambdacost,
        samplecem = samplecem,   
        elitenum = elitenum,     
        cemiter = cemiter,       
        H = T,   #H
        featsize = dpolicyparam,  
        featsize_x = numfeatures, 
        gamma = 1.,   #gamma  
    )
#------------------Only Cumulative Cost-------------------#
@info "Selecting policy parameter"
PolicyMat.W .= reshape(RolloutandSelect(gt_polselect), size(PolicyMat.W,1), size(PolicyMat.W,2))

@info "Running"
reset!(env)
opt = ControllerIterator((action, state, obs) -> getaction!(action, obs, rff_pi_lr, PolicyMat.W),
            env; T = T, plotiter = 50)
for _ in opt # runs iterator
end
@info "cumulative reward" sum(opt.trajectory.rewards)

Odata = opt.trajectory.observations
@info "median value" sqrt(median(pairwise(Euclidean(), Odata, dims=2).^2)/2)
Koopman_gt = rff_x_lr(Odata[:,2:end])*(rff_x_lr(Odata[:,1:end-1]))'/
(rff_x_lr(Odata[:,1:end-1])*(rff_x_lr(Odata[:,1:end-1]))'+I)
Lcost_gt = lambdacost(rff_x_lr(Odata[:,2:end]),rff_x_lr(Odata[:,1:end-1]),arg)
@info "spectrum cost" Lcost_gt
@info "Joint trajectories"

#------------------With Spectrum Cost-------------------#
@info "Selecting policy parameter"
PolicyMat.W .= reshape(RolloutandSelect(sp_polselect), size(PolicyMat.W,1), size(PolicyMat.W,2))

@info "Running"
reset!(env)
opt_sp = ControllerIterator((action, state, obs) -> getaction!(action, obs, rff_pi_lr, PolicyMat.W),
                                 env; T = T, plotiter = 50)
for _ in opt_sp # runs iterator
end
@info "cumulative cost" sum(opt_sp.trajectory.rewards)

Odata_sp = opt_sp.trajectory.observations
Koopman_sp = rff_x_lr(Odata_sp[:,2:end])*(rff_x_lr(Odata_sp[:,1:end-1]))'/
(rff_x_lr(Odata_sp[:,1:end-1])*(rff_x_lr(Odata_sp[:,1:end-1]))'+I)
Lcost_sp = lambdacost(rff_x_lr(Odata_sp[:,2:end]),rff_x_lr(Odata_sp[:,1:end-1]),arg)
@info "spectrum cost" Lcost_sp

#------------------Saving data-------------------#
exper = Experiment("log/data3_"*dfile_num*".jlso", overwrite = true)

exper[:rff_x] = rff_x_lr
exper[:rff_pi] = rff_pi_lr

exper[:Koopman_gt] = Koopman_gt; exper[:Lcost_sp] = Lcost_sp; 
exper[:Lcost_gt] = Lcost_gt;
exper[:opt_sp] = opt_sp; exper[:opt] = opt
exper[:Koopman_sp] = Koopman_sp;
exper[:cumulative_gt] = sum(opt.trajectory.rewards); 
exper[:cumulative_sp] = sum(opt_sp.trajectory.rewards)


finish!(exper);