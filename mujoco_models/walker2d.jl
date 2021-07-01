using LyceumMuJoCo
using UnsafeArrays
using Random
using Shapes
using LyceumBase.Tools
#------------------Constructing Environment-------------------#
struct Walker2d{Sim, OSpace} <: AbstractMuJoCoEnvironment
    sim::Sim
    #statespace::SSpace
    obsspace::OSpace
    #last_torso_x::Float64
    randreset_distribution::Uniform{Float64}
    function Walker2d(sim::MJSim)
        ospace = MultiShape(
            cropped_qpos = VectorShape(Float64, sim.m.nq - 1),
            qvel = VectorShape(Float64, sim.m.nv)
        )
        env = new{typeof(sim), typeof(ospace)}(
            sim, ospace, Uniform(-0.005, 0.005)
        )
        reset!(env)
    end
end

Walker2d() = first(tconstruct(Walker2d, 1))

function LyceumBase.tconstruct(::Type{Walker2d}, n::Integer)
    modelpath = joinpath(@__DIR__, "walker2d.xml")
    return Tuple(Walker2d(s) for s in tconstruct(MJSim, n, modelpath, skip=4))
end


@inline LyceumMuJoCo.getsim(env::Walker2d) = env.sim
@inline LyceumMuJoCo.obsspace(env::Walker2d) = env.obsspace
#@inline LyceumMuJoCo.statespace(env::Walker2d) = env.statespace



function LyceumMuJoCo.getobs!(obs, env::Walker2d)
    checkaxes(obsspace(env), obs)
    qpos = env.sim.d.qpos
    qvel = env.sim.d.qvel
    @views @uviews qpos obs begin
        shaped = obsspace(env)(obs)
        copyto!(shaped.cropped_qpos, qpos[2:end]) 
        shaped.qvel .= clamp.(qvel, -10.0, 10.0) #./ 10.0
    end
     
    obs
end


function LyceumMuJoCo.getreward(state, action, obs, env::Walker2d)
    o = obsspace(env)(obs)
    reward = o.qvel[1] #* 10.0
    reward -= 0.001 * norm(action)^2
    #reward -= 3. * ((o[1] - 1.3)^2)
    reward
end
function LyceumMuJoCo.geteval(state, action, obs, env::Walker2d)
    #o = obsspace(env)(obs)
    #eval = o.qvel[1] #* 10.0
    eval = _torso_x(env)
    eval
end

function LyceumMuJoCo.reset!(env::Walker2d)
    LyceumMuJoCo.reset_nofwd!(env.sim)
    nq, nv = env.sim.m.nq, env.sim.m.nv
    forward!(env.sim)
    env
end

function LyceumMuJoCo.randreset!(env::Walker2d)
    LyceumMuJoCo.reset_nofwd!(env.sim)
    nq, nv = env.sim.m.nq, env.sim.m.nv
    env.sim.d.qpos .+= rand(Uniform(-0.005, 0.005), nq)
    env.sim.d.qvel .+= rand(Uniform(-0.005, 0.005), nv)
    forward!(env.sim)
    env
end



function LyceumMuJoCo.isdone(state, obs, env::Walker2d)
    #height = _torso_height(env)
    #ang = _torso_ang(env)
    #done = not (height > 0.8 and height < 2.0 and
     #               ang > -1.0 and ang < 1.0)
    return false#done
end

@inline _torso_x(env::Walker2d) = env.sim.d.qpos[1]
@inline _torso_height(env::Walker2d) = env.sim.d.qposs[2]
@inline _torso_ang(env::Walker2d) = env.sim.d.qpos[3]
