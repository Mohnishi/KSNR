using Revise
using LinearAlgebra, Random, Statistics
BLAS.set_num_threads(32)
using Plots, JLSO
using LyceumBase.Tools, LyceumBase, LyceumAI, LyceumMuJoCo, MuJoCo, UniversalLogger, Shapes
using FastClosures, Distributions, Parameters
using UnsafeArrays, ElasticArrays
using Distances

using LaTeXStrings
using Plots.PlotMeasures
using Formatting

include("mujoco_models/cartpole_stable.jl")
# # make directories to store log files
_ckmkdir(f) = isdir(f) == false && mkdir(f)
_ckmkdir("log")
_ckmkdir("plot")

# #--------------- execute ---------------------# 
if executescripts
    seednum = 100
    include("scripts/cartpole_sim.jl")
end


#--------------- compute/plot the results ---------------# 

d2 = read("log/data2.jlso", JLSOFile)
# extracting data
Lcost_gt = Float64(d2[:Lcost_gt]); Lcost_sp = Float64(d2[:Lcost_sp]);
cumulative_gt = Float64(d2[:cumulative_gt]); cumulative_sp = Float64(d2[:cumulative_sp]);
opt = d2[:opt]; opt_sp = d2[:opt_sp];
Koopman_gt = d2[:Koopman_gt]; Koopman_sp = d2[:Koopman_sp];

opt1 = opt.trajectory.evaluations; opt2 = opt_sp.trajectory.evaluations
sr_gt = maximum(abs.(eigvals(Koopman_gt)))
sr_sp = maximum(abs.(eigvals(Koopman_sp)))

println("The cumulative reward when NOT using spectrum cost: $cumulative_gt")
println("The spectrum radius when NOT using spectrum cost: $sr_gt")
println("The cumulative reward when using spectrum cost: $cumulative_sp")
println("The spectrum radius when using spectrum cost: $sr_sp")


titles = ["Velocity trajectories" "" ""]
#--------------##--------------##--------------#
## ground truth
println("Saving velocity trajectory.")
pl1 = plot(1:length(opt1), opt1, grid=true, 
xlabel="timesteps", ylabel="velocity", xlim=(0,length(opt1)),
guidefontsize=40, tickfontsize=35, foreground_color_border="gray", 
legend=false,linewidth = 6,linestyle=:dot,linecolor="blue",framestyle=:box, size = (1600, 1000),
bottom_margin=30mm,left_margin=20mm,right_margin=25mm,top_margin=10mm,
yformatter=y->format( y, precision= 1),minorgrid=true)
#--------------#
plot!(1:length(opt2),opt2,  linewidth = 6,linecolor="red", legend=false)
plot!(1:length(opt2),zeros(length(opt2)), linestyle=:dot,  linewidth = 5.5,linecolor="black", legend=false)
#--------------#
plegend1 = plot(1:1,linewidth=4,
linestyle=:dot,linecolor="blue",
legend=:left,label = "no spectrum cost",
framestyle = :none,legendfontsize=30,
foreground_color_legend = nothing)

plegend2 = plot(1:1,linewidth=4,
linestyle=:solid,linecolor="red",
legend=:left,label = "with spectrum cost",
framestyle = :none,legendfontsize=30,
foreground_color_legend = nothing)

#--------------#
plot1 = plot(pl1,plegend1,plegend2,
layout=@layout([a;b{0.1h} c{0.1h}]), size = (1600, 1100),
title=titles, titlefontsize=30)
savefig(plot1, "plot/plot_cartstable.png")