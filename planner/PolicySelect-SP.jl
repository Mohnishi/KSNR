"""
This file is for selecting policy.  Ground truth model with reward and spectrum.
Base code borrowed from LyceumAI.jl https://github.com/Lyceum/LyceumAI.jl
    envs: these are for parallel computing of shootings (CEM).  In our work, these envs represent learned models.
"""

using StaticArrays, Kronecker
using Base: @propagate_inbounds, require_one_based_indexing

const AbsVec = AbstractVector
const AbsMat = AbstractMatrix

struct PolicySelectSP{DT<:AbstractFloat,Env,P,X,A,LC,Obs,State}
    # PolicySelectSP parameters
    samplecem::Int
    elitenum::Int
    cemiter::Int
    H::Int
    gamma::DT
    envs::Vector{Env} # one per thread
    baseenv::Env
    phi_pol::P
    phi_x::X
    arg::A
    lambdacost::LC
    # internal
    stdpolicy::Matrix{DT}
    meanpolicy::Matrix{DT}
    toppolicy::Matrix{DT}
    policyparam::Array{DT,3}
    costs::Vector{DT}
    obsbuffers::Vector{Obs}
    statebuffers::Vector{State}
    featbuffers::Vector{Matrix}
    actsize::Int
    featsize::Int
    featsize_x::Int

    function PolicySelectSP{DT}(
        env_tconstructor,
        phi_pol,
        phi_x,
        arg,
        lambdacost,
        samplecem::Integer,
        elitenum::Integer,
        cemiter::Integer,
        H::Integer,
        featsize::Integer,
        featsize_x::Integer,
        gamma::Real,
    ) where {DT<:AbstractFloat}
        envs = [e for e in env_tconstructor(Threads.nthreads())]
        baseenv = first(env_tconstructor(1))
        ssp = statespace(first(envs))
        osp = obsspace(first(envs))
        actsize = length(actionspace(first(envs)))
        Policyparamsize = actsize * featsize

        samplecem > 0 || error("samplecem must be > 0. Got $samplecem.")
        elitenum > 0 || error("samplecem must be > 0. Got $elitenum.")
        cemiter > 0 || error("cemiter must be > 0. Got $cemiter.")
        H > 1 || error("H must be > 1. Got $H.")
        featsize_x > 0 || error("feature size must be > 0. Got $featsize_x.")
        Policyparamsize > 0 || error("Policyparamsize must be > 0. Got $v.")
        0 < gamma <= 1 || throw(ArgumentError("gamma must be in interval (0, 1]"))
        meanpolicy = zeros(DT, 1,Policyparamsize)
        stdpolicy = ones(DT, 1,Policyparamsize)
        toppolicy = ones(DT, 1,Policyparamsize)
        policyparam = zeros(DT, 1,Policyparamsize, samplecem)
        costs = zeros(DT, samplecem)
        obsbuffers = [allocate(osp) for _ = 1:Threads.nthreads()]
        statebuffers = [allocate(ssp) for _ = 1:Threads.nthreads()]
        featbuffers = [zeros(DT, featsize_x, H) for _ = 1:Threads.nthreads()]

        new{
            DT,
            eltype(envs),
            typeof(phi_pol),
            typeof(phi_x),
            typeof(arg),
            typeof(lambdacost),
            eltype(obsbuffers),
            eltype(statebuffers),
        }(
            samplecem,
            elitenum,
            cemiter,
            H,
            gamma,
            envs,
            baseenv,
            phi_pol,
            phi_x,
            arg,
            lambdacost,
            stdpolicy,
            meanpolicy,
            toppolicy,
            policyparam,
            costs,
            obsbuffers,
            statebuffers,
            featbuffers,
            actsize,
            featsize,
            featsize_x
        )
       
    end
end


function PolicySelectSP{DT}(;
    env_tconstructor,
    phi_pol,
    phi_x,
    arg,
    lambdacost,
    samplecem,
    elitenum,
    cemiter,
    H,
    featsize,
    featsize_x,
    gamma = 1,
) where {DT<:AbstractFloat}
    PolicySelectSP{DT}(env_tconstructor, phi_pol, phi_x, arg, lambdacost, 
    samplecem, elitenum, cemiter, H, featsize, featsize_x, gamma)
end

PolicySelectSP(args...; kwargs...) = PolicySelectSP{Float32}(args...; kwargs...)

function LyceumBase.reset!(pol::PolicySelectSP{DT}) where {DT}
    pol.meanpolicy .= 0.
    pol.stdpolicy .= 1.
    pol.toppolicy .= 0.
    pol.policyparam .= 0.
    pol.costs .= 0.
end


function RolloutandSelect(
    pol::PolicySelectSP{DT};
    nthreads::Integer = Threads.nthreads(),
) where {DT}
    nthreads = min(pol.samplecem, nthreads)
    for i=1:pol.cemiter
        if i%5 == 0; @info "iter $i"; end;
        sampler = Normal.(pol.meanpolicy, pol.stdpolicy)
        for s = 1:pol.samplecem
            pol.policyparam[:,:,s]  =  rand.(sampler)
        end
        randreset!(pol.baseenv)
      
        if nthreads == 1
            # short circuit
            rolloutselect!(pol, 1:pol.samplecem, sampler)
        else
            kranges = splitrange(pol.samplecem, nthreads)
            @sync for i = 1:nthreads
                Threads.@spawn rolloutselect!(pol, kranges[i], sampler)
            end
        end
        
        eliteind = sortperm(pol.costs, rev=false)
        pol.meanpolicy .= mean(pol.policyparam[:,:,eliteind[1:pol.elitenum]], dims=3)[:,:,1]
        pol.stdpolicy .= std(pol.policyparam[:,:,eliteind[1:pol.elitenum]], dims=3)[:,:,1]
        pol.toppolicy .= pol.policyparam[:,:,eliteind[1]]
    end

    return pol.toppolicy
    
end

function rolloutselect!(pol::PolicySelectSP{DT}, krange, sampler) where {DT}
    tid = Threads.threadid()
    for k in krange
        env = pol.envs[tid]
        obsbuf = pol.obsbuffers[tid]
        statebuf = pol.statebuffers[tid]
        getstate!(statebuf, pol.baseenv)
        setstate!(env, statebuf)
        getobs!(obsbuf, pol.baseenv)
        pol.featbuffers[tid][:,1] = pol.phi_x(obsbuf)
        discountedreward = zero(DT)
        discountfactor = one(DT)
        policyTheta = reshape(pol.policyparam[:,:,k], pol.actsize, pol.featsize)
        for t = 1:pol.H-1
            action_t = policyTheta * pol.phi_pol(obsbuf) #.+= 0.01*randn(pol.actsize)
            setaction!(env, action_t)
            step!(env)
            getobs!(obsbuf, env)
            getstate!(statebuf, env)
            pol.featbuffers[tid][:,t+1] = pol.phi_x(obsbuf)
            reward = getreward(statebuf, action_t, obsbuf, env)
            discountedreward += reward * discountfactor
            discountfactor *= pol.gamma
        end 

        pol.costs[k] = - discountedreward + pol.lambdacost(pol.featbuffers[tid][:,2:end],pol.featbuffers[tid][:,1:end-1],pol.arg) 
    end
end

