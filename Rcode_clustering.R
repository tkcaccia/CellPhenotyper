# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Extract the two strings
output <- args[1]

library(KODAMA)
library(KODAMAextra)
library(SPARK)
library(data.table)
library(irlba)
library(bluster)
library(igraph)

txt=paste(output,"/kodama_full_20.RData",sep="")

# save(vis,vis_tile,vis_nuclei,xy,file=txt)
load(file=txt)


g <- makeSNNGraph(as.matrix(vis), k = 100)
clu = cluster_louvain(g, resolution = 0.2)
t = clu$membership

da=data.frame(label=rownames(vis),cluster=t)

txt=paste(output,"/cluster.csv",sep="")

write.csv(da,row.names = FALSE,txt)






