using Revise
using LinearAlgebra, Random, Statistics
BLAS.set_num_threads(32)
using Plots, JLSO
using LyceumBase.Tools, LyceumBase, LyceumAI, UniversalLogger, Shapes
using FastClosures, Distributions, Parameters
using UnsafeArrays, ElasticArrays
using Distances

using LaTeXStrings
using Plots.PlotMeasures
using Formatting

# make directories to store log files
_ckmkdir(f) = isdir(f) == false && mkdir(f)
_ckmkdir("log")
_ckmkdir("plot")

#--------------- execute ---------------------# 
if executescripts
	seednum = 100
	include("scripts/singleint_sim.jl")
end


#--------------- compute/plot the results ---------------# 

d1 = read("log/data1.jlso", JLSOFile)
# extracting data

Odata = Matrix(d1[:Odata]); Odata_sp = Matrix(d1[:Odata_sp]);
Odata_sp2 = Matrix(d1[:Odata_sp2]); xdata_t = Vector(d1[:xdata_t]);
ydata_t = Vector(d1[:ydata_t]); xdata = Vector(d1[:xdata]);
ydata = Vector(d1[:ydata]); xdata2 = Vector(d1[:xdata2]);
ydata2 = Vector(d1[:ydata2]);

titles = ["Observation trajectories" "X-Y trajectory" "" "" ""]
#--------------##--------------##--------------#
## ground truth
println("Saving trajectories of the ground-truth dynamics.")
pl1 = plot(1:size(Odata,2), Odata', grid=true, 
xlabel="timesteps", ylabel="observations", xlim=(0,size(Odata,2)),
guidefontsize=40, tickfontsize=35, foreground_color_border="gray", 
legend=false,linewidth = 4,linestyle=[:solid :solid :solid],framestyle=:box, size = (1200, 1000),
bottom_margin=50mm,left_margin=20mm,right_margin=25mm,top_margin=10mm,
yformatter=y->format( y, precision= 1),minorgrid=true)
#--------------#
pl1s = plot(xdata_t, ydata_t, grid=true,
xlabel=L"x", ylabel=L"y", 
guidefontsize=40, tickfontsize=35, foreground_color_border="gray", 
legend=false, linewidth = 4,linecolor = "black",framestyle=:box, size = (1200, 1000),
bottom_margin=50mm,left_margin=20mm,right_margin=25mm,top_margin=10mm, minorgrid=true,
yformatter=y->format( y, precision= 1))
#--------------#
plegend1 = plot(1:1,linecolor=1,linewidth=4,legend=:left,label = L"r~\mathrm{(distance)}",framestyle = :none,legendfontsize=50,
foreground_color_legend = nothing)
plegend2 = plot(1:1,linecolor=2,linewidth=4,legend=:left,label = L"\cos(\theta)",framestyle = :none,legendfontsize=50,
foreground_color_legend = nothing)
plegend3 = plot(1:1,linecolor=3,linewidth=4,legend=:left,label = L"\sin(\theta)",framestyle = :none,legendfontsize=50,
foreground_color_legend = nothing)
#--------------#
plot1 = plot(pl1,pl1s,plegend1,plegend2,plegend3,
layout=@layout([a b;c{0.1h} d{0.1h} e{0.1h}]), size = (2400, 1200),
title=titles, titlefontsize=30)
savefig(plot1, "plot/plot_singleint_1.png")
#--------------##--------------##--------------#
## top mode
println("Saving trajectories of the top-mode imitating dynamics.")
pl2 = plot(1:size(Odata_sp,2), Odata_sp', grid=true, 
xlabel="timesteps", ylabel="observations", xlim=(0,size(Odata_sp,2)), 
guidefontsize=40, tickfontsize=35, foreground_color_border="gray", 
legend=false,linewidth = 4,linestyle=[:solid :solid :solid],framestyle=:box, size = (1200, 1000),
bottom_margin=50mm,left_margin=20mm,right_margin=25mm,top_margin=10mm,
yformatter=y->format( y, precision= 1),minorgrid=true)
#--------------#
pl2s = plot(xdata, ydata, grid=true,
xlabel=L"x", ylabel=L"y", 
guidefontsize=40, tickfontsize=35, foreground_color_border="gray", 
legend=false, linewidth = 4,linecolor = "black",framestyle=:box, size = (1200, 1000),
bottom_margin=50mm,left_margin=20mm,right_margin=25mm,top_margin=10mm, minorgrid=true,
yformatter=y->format( y, precision= 1))
#--------------#
plot2 = plot(pl2,pl2s,plegend1,plegend2,plegend3,
layout=@layout([a b;c{0.1h} d{0.1h} e{0.1h}]), size = (2400, 1200),
title=titles, titlefontsize=30)
savefig(plot2, "plot/plot_singleint_2.png")
#--------------##--------------##--------------#
## frobenius
println("Saving trajectories of the Frobenius imitating dynamics.")
pl3 = plot(1:size(Odata_sp2,2), Odata_sp2', grid=true, 
xlabel="timesteps", ylabel="observations", xlim=(0,size(Odata_sp2,2)), 
guidefontsize=40, tickfontsize=35, foreground_color_border="gray", 
legend=false,linewidth = 4,linestyle=[:solid :solid :solid],framestyle=:box, size = (1200, 1000),
bottom_margin=50mm,left_margin=20mm,right_margin=25mm,top_margin=10mm,
yformatter=y->format( y, precision= 1),minorgrid=true)
#--------------#
pl3s = plot(xdata2, ydata2, grid=true,
xlabel=L"x", ylabel=L"y", 
guidefontsize=40, tickfontsize=35, foreground_color_border="gray", 
legend=false, linewidth = 4,linecolor = "black",framestyle=:box, size = (1200, 1000),
bottom_margin=50mm,left_margin=20mm,right_margin=25mm,top_margin=10mm, minorgrid=true,
yformatter=y->format( y, precision= 1))
#--------------#
plot3 = plot(pl3,pl3s,plegend1,plegend2,plegend3,
layout=@layout([a b;c{0.1h} d{0.1h} e{0.1h}]), size = (2400, 1200),
title=titles, titlefontsize=30)
savefig(plot3, "plot/plot_singleint_3.png")


