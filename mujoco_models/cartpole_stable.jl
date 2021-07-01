using LyceumMuJoCo
using UnsafeArrays
using Random
using Shapes
using Statistics
#------------------Constructing Environment-------------------#

include("reward.jl")

struct CartpoleStable{S<:MJSim,O<:MultiShape} <: AbstractMuJoCoEnvironment
    sim::S
    obsspace::O
    function CartpoleStable(sim::MJSim)
        obsspace = MultiShape(
            pos = MultiShape(
                #cart = ScalarShape(Float64),
                pole_zz = ScalarShape(Float64),
                pole_xz = ScalarShape(Float64),
            ),
            vel = MultiShape(
                cart = ScalarShape(Float64),
                pole_ang = ScalarShape(Float64),
            ),
        )
        env = new{typeof(sim), typeof(obsspace)}(sim, obsspace)
        LyceumMuJoCo.reset!(env)
    end
end

LyceumMuJoCo.getsim(env::CartpoleStable) = env.sim
LyceumMuJoCo.obsspace(env::CartpoleStable) = env.obsspace

CartpoleStable() = first(tconstruct(CartpoleStable, 1))
function LyceumBase.tconstruct(::Type{CartpoleStable}, n::Integer)
    modelpath = joinpath(@__DIR__, "cartpole_stable.xml")
    Tuple(CartpoleStable(s) for s in tconstruct(MJSim, n, modelpath, skip=1))
end

function LyceumMuJoCo.setstate!(env::CartpoleStable, state)
    sim = env.sim
    LyceumMuJoCo.copystate!(sim, state)
    forward!(sim)
    shaped = statespace(sim)(state)
    @uviews shaped begin copyto!(sim.d.qacc_warmstart, shaped.qacc_warmstart) end
    sim
end

# overwriting some of the functions for the environment
function LyceumMuJoCo.isdone(state, action, obs, env::CartpoleStable)
    return false
end

# initializing mujoco
function LyceumMuJoCo.reset!(env::CartpoleStable)
    LyceumMuJoCo.reset_nofwd!(env.sim)
    qpos = env.sim.dn.qpos
    @uviews qpos begin
        qpos[:hinge_1] = 0
    end
    forward!(env.sim)
    env
end

function LyceumMuJoCo.randreset!(rng::Random.AbstractRNG, env::CartpoleStable)
    LyceumMuJoCo.reset_nofwd!(env.sim)

    qpos = env.sim.dn.qpos
    @uviews qpos begin
        qpos[:hinge_1] = 0
    end
    @uviews qpos begin
        qpos[:slider] = 0.01 * randn()
        qpos[:hinge_1] = 0 + 0.01 * randn()
    end
    randn!(env.sim.d.qvel)
    forward!(env.sim)
    env
end


function LyceumMuJoCo.getobs!(obs, env::CartpoleStable)
    @uviews obs begin
        sobs = obsspace(env)(obs)
        sobs.pos.pole_zz = env.sim.dn.xmat[:z, :z, :pole_1] 
        sobs.pos.pole_xz = env.sim.dn.xmat[:x, :z, :pole_1]
        sobs.vel .= env.sim.d.qvel 
    end
    obs
end

function LyceumMuJoCo.geteval(state, action, obs, env::CartpoleStable)
    env.sim.dn.qvel[1]
end


function LyceumMuJoCo.getreward(state, action, obs, env::CartpoleStable)
    sobs = obsspace(env)(obs)
    upright = 0
    if sobs.pos.pole_zz < 0
         upright = -1.
    end
    vel = abs(sobs.vel.cart)
    upright + 0.001 * vel
end
