"""
This file is for selecting policy.  Learned model with reward and spectrum.
Base code borrowed from LyceumAI.jl https://github.com/Lyceum/LyceumAI.jl
    envs: these are for parallel computing of shootings (CEM).  In our work, these envs represent learned models.
"""

using StaticArrays, Kronecker
using Base: @propagate_inbounds, require_one_based_indexing

const AbsVec = AbstractVector
const AbsMat = AbstractMatrix

struct PolicySelect{DT<:AbstractFloat,Env,F,L,A,Obs,State}
    # PolicySelect parameters
    samplecem::Int
    elitenum::Int
    cemiter::Int
    H::Int
    gamma::DT
    envs::Vector{Env} # one per thread
    baseenv::Env
    Mmat::AbstractArray{DT}
    phi_op::F
    lambdacost::L
    arg::A
    # internal
    stdpolicy::Matrix{DT}
    meanpolicy::Matrix{DT}
    toppolicy::Matrix{DT}
    policyparam::Array{DT,3}
    costs::Vector{DT}
    obsbuffers::Vector{Obs}
    statebuffers::Vector{State}

    function PolicySelect{DT}(
        env_tconstructor,
        Mmat::AbstractArray{DT},
        phi_op,
        lambdacost,
        arg,
        samplecem::Integer,
        elitenum::Integer,
        cemiter::Integer,
        H::Integer,
        Policyparamsize::Integer,
        gamma::Real,
    ) where {DT<:AbstractFloat}
        envs = [e for e in env_tconstructor(Threads.nthreads())]
        baseenv = first(env_tconstructor(1))
        ssp = statespace(first(envs))
        osp = obsspace(first(envs))

        samplecem > 0 || error("samplecem must be > 0. Got $samplecem.")
        elitenum > 0 || error("samplecem must be > 0. Got $elitenum.")
        cemiter > 0 || error("cemiter must be > 0. Got $cemiter.")
        H > 1 || error("H must be > 1. Got $H.")
        Policyparamsize > 0 || error("Policyparamsize must be > 0. Got $v.")
        0 < gamma <= 1 || throw(ArgumentError("gamma must be in interval (0, 1]"))
        meanpolicy = zeros(DT, 1,Policyparamsize)
        stdpolicy = ones(DT, 1,Policyparamsize)
        toppolicy = ones(DT, 1,Policyparamsize)
        policyparam = zeros(DT, 1,Policyparamsize, samplecem)
        costs = zeros(DT, samplecem)
        obsbuffers = [allocate(osp) for _ = 1:Threads.nthreads()]
        statebuffers = [allocate(ssp) for _ = 1:Threads.nthreads()]
        
        new{
            DT,
            eltype(envs),
            typeof(phi_op),
            typeof(lambdacost),
            typeof(arg),
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
            Mmat,
            phi_op,
            lambdacost,
            arg,
            stdpolicy,
            meanpolicy,
            toppolicy,
            policyparam,
            costs,
            obsbuffers,
            statebuffers
        )
       
    end
end

#using DocStringExtensions

function PolicySelect{DT}(;
    env_tconstructor,
    Mmat,
    phi_op,
    lambdacost,
    arg,
    samplecem,
    elitenum,
    cemiter,
    H,
    Policyparamsize,
    gamma = 1
) where {DT<:AbstractFloat}
    PolicySelect{DT}(env_tconstructor, Mmat, phi_op, lambdacost,
    arg, samplecem, elitenum,
    cemiter, H, Policyparamsize, gamma)
end

PolicySelect(args...; kwargs...) = PolicySelect{Float32}(args...; kwargs...)

function LyceumBase.reset!(pol::PolicySelect{DT}) where {DT}
    pol.meanpolicy .= 0.
    pol.stdpolicy .= 1.
    pol.toppolicy .= 0.
    pol.policyparam .= 0.
    pol.costs .= 0.
end


function RolloutandSelect(
    pol::PolicySelect{DT};
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

function rolloutselect!(pol::PolicySelect{DT}, krange, sampler) where {DT}
    tid = Threads.threadid()
    for k in krange
        env = pol.envs[tid]
        obsbuf = pol.obsbuffers[tid]
        statebuf = pol.statebuffers[tid]
        getstate!(statebuf, pol.baseenv)
        setstate!(env, statebuf)
        getobs!(obsbuf, pol.baseenv)
        KoopmanMat = reduce(hcat, [pol.Mmat[:,:,i]*pol.phi_op(Matrix(pol.policyparam[:,:,k]')) for 
        i in 1:size(pol.Mmat,3)])
        env.KoopmanMat .= KoopmanMat
        discountedreward = zero(DT)
        discountfactor = one(DT)
        for t = 1:pol.H-1
            step!(env)
            getobs!(obsbuf, env)
            getstate!(statebuf, env)
            reward = getreward(obsbuf, env)
            discountedreward += reward * discountfactor
            discountfactor *= pol.gamma
        end 

        pol.costs[k] = - discountedreward + pol.lambdacost(KoopmanMat, pol.arg) +
        100 * max(2., maximum(abs.(pol.policyparam[:,:,k])))
    end
end



