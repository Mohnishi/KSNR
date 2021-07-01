"""
This file is for executing single integrator task with CEM search.
Using nonlinear feadback -> recreating limit cycle by
imitating Koopman spectrum.
"""

#------------------Packages-------------------#
using LinearAlgebra, Random, Statistics
BLAS.set_num_threads(32)
using Plots, JLSO
using ElasticArrays
using LyceumBase.Tools, LyceumBase, LyceumAI, UniversalLogger, Shapes
using FastClosures, Distributions, Parameters
using UnsafeArrays
using Distances


include("../utils/weightmat.jl")
include("../models/singleint.jl")
include("../planner/PolicySelect-GT.jl")
include("../planner/PolicySelect-SP.jl")

#-----------------------Constants -------------------#
numfeatures = 80      # dimH  (for state x)
T           = 80      # task horizon (H in the note)
samplecem   = 200      # sample number of CEM every iteration
elitenum    = 20       # Elite number of CEM
cemiter     = 50        # number of iterations for CEM

trainiter   = 500

@info "SingleIntegrator Task"
@info "Seed number" seednum
Random.seed!(seednum)
#----------------------Environment-------------------#
mjenvs = tconstruct(SingleIntegrator, Threads.nthreads());
env = SingleIntegrator()
dobs, dact = length(obsspace(env)), length(actionspace(env))
dpolicyparam = 50       # dim Theta: dimension for policy
 
PolicyMat  = PredictionMat(dpolicyparam, dact)  # Policy is Theata pi(x) and this is pi
KoopmanMat = PredictionMat(numfeatures, numfeatures)

@info "Dimensions of observations/actions/outputs" dobs dact 
@info "Simulation timestep:" LyceumBase.timestep(env)
#----------------------Functions-------------------#
# Frobenius norm
function lambdacost2(Y::AbstractArray,X::AbstractArray,arg)
    sum((arg - (Y * X' / (X * X' + I) ) ).^2 )
end
# Top mode
function lambdacost(Y::AbstractArray,X::AbstractArray,arg)
    K = eigen(Y * X' / (X * X' + I))
    ind = sortperm(abs.(K.values), rev=true)
    vec = K.vectors[:,ind[1]]
    sum(abs.(arg - vec))
end
# target trajectory
function getaction_training!(
    action::AbstractVector,
    obs
) 
    action[1] = obs[1] * (1-obs[1]^2)
    action[2] = 1
    return action
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
# transforming xy to polar
function transx2p(x::AbstractFloat,y::AbstractFloat)
    r = sqrt(x^2+y^2)
    costheta = x / r
    sintheta = y / r
    return [r; costheta; sintheta]
end
# transforming polar to xy
function transp2x(r,costheta,sintheta)
    x = r .* costheta
    y = r .* sintheta
    return x, y
end


#------------------Define Features-------------------#
include("../utils/rff.jl")
rffbandwidth_x = 3.
rffbandwidth_pi = 2.
const rff_x = RandomFourierFunctions{Float64}(rffbandwidth_x, dobs, numfeatures)
const rff_pi = RandomFourierFunctions{Float64}(rffbandwidth_pi, dobs, dpolicyparam)

#----------------------Environment-------------------#
env_tconstructor = n -> tconstruct(SingleIntegrator, n)

#------------------------- Training------------------#

Odata_trainX = ElasticArray{Float64}(undef,dobs,0)
Odata_trainY = ElasticArray{Float64}(undef,dobs,0)
for i=1:trainiter
    randreset!(env)
    train = ControllerIterator((action, state, obs) -> getaction_training!(action, obs),
                env; T = T, plotiter = 500)
    for _ in train # runs iterator
    end
    Odata_train = train.trajectory.observations
    append!(Odata_trainX, Odata_train[:,1:end-1])
    append!(Odata_trainY, Odata_train[:,2:end])
end
Odata_trainX = Matrix(Odata_trainX); Odata_trainY = Matrix(Odata_trainY)
Koop_targ = rff_x(Odata_trainY)*(rff_x(Odata_trainX))'/
(rff_x(Odata_trainX)*(rff_x(Odata_trainX))'+I)
###---------------------------###
@info "Running"
reset!(env)
opt = ControllerIterator((action, state, obs) -> getaction_training!(action, obs),
                                 env; T = T*3, plotiter = 40)
for _ in opt # runs iterator
end
Odata = opt.trajectory.observations
###---------------------------###
eig_t = eigen(Koop_targ)
ind = sortperm(abs.(eig_t.values), rev=true)
vec = eig_t.vectors[:,ind[1]]

xdata_t, ydata_t = transp2x(Odata[1,:],Odata[2,:],Odata[3,:])
# for later use : args
arg = vec
arg2 = Koop_targ
#------------------- Koopman cost Top mode------------------#
sp_polselect = PolicySelectSP(
        env_tconstructor = n -> tconstruct(SingleIntegrator, n),
        phi_pol = rff_pi, 
        phi_x  =  rff_x,  
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
###---------------------------###
@info "Selecting policy parameter"
PolicyMat.W .= reshape(RolloutandSelect(sp_polselect), size(PolicyMat.W,1), size(PolicyMat.W,2))
###---------------------------###
@info "Running"
reset!(env)
opt_sp = ControllerIterator((action, state, obs) -> getaction!(action, obs, rff_pi, PolicyMat.W),
                                 env; T = T*3, plotiter = 40)
for _ in opt_sp # runs iterator
end
###---------------------------###
Odata_sp = opt_sp.trajectory.observations
Koopman_sp = rff_x(Odata_sp[:,2:end])*(rff_x(Odata_sp[:,1:end-1]))'/
(rff_x(Odata_sp[:,1:end-1])*(rff_x(Odata_sp[:,1:end-1]))'+I)
Lcost_sp = lambdacost(rff_x(Odata_sp[:,2:end]),rff_x(Odata_sp[:,1:end-1]),arg)
@info "spectrum cost" Lcost_sp
###---------------------------###
eig_sp = eigen(Koopman_sp)
ind_sp = sortperm(abs.(eig_sp.values), rev=true)
vec_sp = eig_sp.vectors[:,ind_sp[1]]
###---------------------------###
xdata, ydata = transp2x(Odata_sp[1,:],Odata_sp[2,:],Odata_sp[3,:])

#------------------- Koopman cost Frobenius------------------#
sp_polselect2 = PolicySelectSP(
        env_tconstructor = n -> tconstruct(SingleIntegrator, n),
        phi_pol = rff_pi, 
        phi_x  =  rff_x, 
        arg = arg2,     
        lambdacost = lambdacost2,
        samplecem = samplecem,   
        elitenum = elitenum,     
        cemiter = cemiter,       
        H = T,   #H
        featsize = dpolicyparam,  
        featsize_x = numfeatures,
        gamma = 1.,   #gamma  
    )
###---------------------------###
@info "Selecting policy parameter"
PolicyMat.W .= reshape(RolloutandSelect(sp_polselect2), size(PolicyMat.W,1), size(PolicyMat.W,2))
###---------------------------###
@info "Running"
reset!(env)
opt_sp2 = ControllerIterator((action, state, obs) -> getaction!(action, obs, rff_pi, PolicyMat.W),
                                 env; T = T*3, plotiter = 40)
for _ in opt_sp2 # runs iterator
end
###---------------------------###
Odata_sp2 = opt_sp2.trajectory.observations
Koopman_sp2 = rff_x(Odata_sp2[:,2:end])*(rff_x(Odata_sp2[:,1:end-1]))'/
(rff_x(Odata_sp2[:,1:end-1])*(rff_x(Odata_sp2[:,1:end-1]))'+I)
Lcost_sp2 = lambdacost2(rff_x(Odata_sp2[:,2:end]),rff_x(Odata_sp2[:,1:end-1]),arg2)
@info "spectrum cost" Lcost_sp2
###---------------------------###
eig_sp2 = eigen(Koopman_sp2)
ind_sp2 = sortperm(abs.(eig_sp2.values), rev=true)
vec_sp2 = eig_sp2.vectors[:,ind_sp2[1]]
###---------------------------###
xdata2, ydata2 = transp2x(Odata_sp2[1,:],Odata_sp2[2,:],Odata_sp2[3,:])

#---------------------saving data--------------------#
# save to file
exper = Experiment("log/data1.jlso", overwrite = true)
exper[:rff_x] = rff_x
exper[:rff_pi] = rff_pi

exper[:ind] = ind; exper[:vec] = vec; 
exper[:Koop_targ] = Koop_targ; exper[:Odata] = Odata;
exper[:xdata_t] = xdata_t; exper[:ydata_t] = ydata_t

exper[:ind_sp] = ind_sp; exper[:vec_sp] = vec_sp; 
exper[:Odata_sp] = Odata_sp; 
exper[:Koopman_sp] = Koopman_sp;
exper[:xdata] = xdata; exper[:ydata] = ydata

exper[:ind_sp2] = ind_sp2; exper[:vec_sp2] = vec_sp2; 
exper[:Odata_sp2] = Odata_sp2; 
exper[:Koopman_sp2] = Koopman_sp2;
exper[:xdata2] = xdata2; exper[:ydata2] = ydata2
finish!(exper);
