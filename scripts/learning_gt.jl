#------------------Policy Select-------------------#
@info "Running ground-truth experiments"
include("../planner/PolicySelect-SP.jl")
include("../planner/PolicySelect-GT.jl")
T           = 500      # task horizon (H in the note)
samplecem   = 200      # sample number of CEM every iteration
elitenum    = 20       # Elite number of CEM
cemiter     = 50        # number of iterations for CEM
function lambdacost(Y::AbstractArray, X::AbstractArray,arg)
        svdx = svd(X)
        U = svdx.U
        V = svdx.V
        S = diagm(svdx.S)
        K = U'*Y*V*S/(S.^2+I)
        sum((arg[1:4,:]-K[1:4,:]).^2) + 
        0.01 * sum(abs.(eigvals(K)))
end
function getaction!(
    action::AbstractVector,
    obs,
    featurize_pi,
    PolicyMatW::AbstractArray;
) 
    # test = copy(PolicyMatW)
    # test .= 0.
    # test[1] = 1.
    # mul!(action, test, featurize_pi(obs))
    mul!(action, PolicyMatW, featurize_pi(obs))
    return action
end
##------ struct--------###
@info "Seed number" seednum
Random.seed!(seednum)
PolicyMat  = PredictionMat(dpolicyparam, dact)  # Policy is Theata pi(x) and this is pi

gt_polselect = PolicySelectGT(
        env_tconstructor = n -> tconstruct(CartpoleGym, n),
        phi_pol = policyfunc,      
        samplecem = samplecem,   
        elitenum = elitenum,     
        cemiter = cemiter,       
        H = T,   #H
        featsize = dpolicyparam,   
        gamma = 1.,   #gamma  
    )

sp_polselect = PolicySelectSP(
        env_tconstructor = n -> tconstruct(CartpoleGym, n),
        phi_pol = policyfunc, 
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

#------------------Only Cumulative cost-------------------#
@info "Selecting policy parameter"
PolicyMat.W .= reshape(RolloutandSelect(gt_polselect), size(PolicyMat.W,1), size(PolicyMat.W,2))

@info "Running"
reset!(env)
opt = ControllerIterator((action, state, obs) -> getaction!(action, obs, policyfunc, PolicyMat.W),
            env; T = T, plotiter = 50)
for _ in opt # runs iterator
end
@info "cumulative reward" sum(opt.trajectory.rewards)

Odata = opt.trajectory.observations
@info "median value" sqrt(median(pairwise(Euclidean(), Odata, dims=2).^2)/2)
Koopman_gt = rff_x(Odata[:,2:end])*(rff_x(Odata[:,1:end-1]))'/
(rff_x(Odata[:,1:end-1])*(rff_x(Odata[:,1:end-1]))'+I)
Lcost_gt = lambdacost(rff_x(Odata[:,2:end]),rff_x(Odata[:,1:end-1]),arg)
@info "spectrum cost" Lcost_gt
@info "policy" PolicyMat.W

#------------------With Koopman cost-------------------#
@info "Selecting policy parameter"
PolicyMat.W .= reshape(RolloutandSelect(sp_polselect), size(PolicyMat.W,1), size(PolicyMat.W,2))

@info "Running"
reset!(env)
opt_sp = ControllerIterator((action, state, obs) -> getaction!(action, obs, policyfunc, PolicyMat.W),
                                 env; T = T, plotiter = 50)
for _ in opt_sp # runs iterator
end
@info "cumulative reward" sum(opt_sp.trajectory.rewards)

Odata_sp = opt_sp.trajectory.observations
Koopman_sp = rff_x(Odata_sp[:,2:end])*(rff_x(Odata_sp[:,1:end-1]))'/
(rff_x(Odata_sp[:,1:end-1])*(rff_x(Odata_sp[:,1:end-1]))'+I)
Lcost_sp = lambdacost(rff_x(Odata_sp[:,2:end]),rff_x(Odata_sp[:,1:end-1]),arg)
@info "spectrum cost" Lcost_sp
@info "policy" PolicyMat.W

#------------------Savind data-------------------#

exper = Experiment("log/data4gt_"*dfile_num*".jlso", overwrite = true)

exper[:Koopman_gt] = Koopman_gt; exper[:Lcost_sp] = Lcost_sp; 
exper[:Lcost_gt] = Lcost_gt;
exper[:opt_sp] = opt_sp; exper[:opt] = opt
exper[:Koopman_sp] = Koopman_sp;
exper[:cumulative_gt] = sum(opt.trajectory.rewards); 
exper[:cumulative_sp] = sum(opt_sp.trajectory.rewards)

finish!(exper); # flushes everything to disk