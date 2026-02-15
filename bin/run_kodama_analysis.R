# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Extract the two strings
dinov2 <- args[1]
annot <- args[2]
output <- args[3]
uni2_folder = args[1]

library(KODAMA)
library(KODAMAextra)
library(SPARK)
library(data.table)
library("irlba")

# sintx -n 4 -N 1
#singularity exec /scratch/firenze/singularity/singularity-r_4.4.0.sif R

#annot = "out_stardist_roi/objects_assigned.csv"
#output="output/KODAMA/"
#uni2_folder="embeddings_uni2_tile"

if(TRUE){

an = read.csv(annot,row.names=1)


ll=list.files( uni2_folder,full.names=TRUE,recursive=TRUE)
r_tile=NULL
if (length(ll) > 0) {
  for (ii in seq_along(ll)) {
    if (!is.na(ll[ii]) && endsWith(ll[ii], "csv.gz")) {
      df <- fread(ll[ii])
      r_tile <- rbind(r_tile, df)
    }
  }
}
if (is.null(r_tile) || nrow(r_tile) == 0) {
  stop(paste("No UNI-2 tile embedding files found in", uni2_folder))
}


ll=list.files("embeddings_uni2_cyto",full.names=TRUE,recursive=TRUE)
r_nuclei=NULL
if (length(ll) > 0) {
  for (ii in seq_along(ll)) {
    if (!is.na(ll[ii]) && endsWith(ll[ii], "csv.gz")) {
      df <- fread(ll[ii])
      r_nuclei <- rbind(r_nuclei, df)
    }
  }
}
if (is.null(r_nuclei) || nrow(r_nuclei) == 0) {
  message("No cytoplasm embedding files found; reusing tile embeddings.")
  r_nuclei <- copy(r_tile)
}


a=as.character(as.vector(r_tile[,"cell_id"])$cell_id)
rownames(r_tile)=a
b=as.character(as.vector(r_nuclei[,"cell_id"])$cell_id)
rownames(r_nuclei)=b
a=intersect(a,b)

r_tile=r_tile[cell_id %chin% a,-c(1:13)]
r_nuclei=r_nuclei[cell_id %chin% a,-c(1:13)]

r_tile <- r_tile[, .SD, .SDcols = patterns("^feat")]
r_nuclei <-r_nuclei[, .SD, .SDcols = patterns("^feat")]


r_tile=as.matrix(r_tile)
r_nuclei=as.matrix(r_nuclei)

rownames(r_tile)=a
rownames(r_nuclei)=a

an=an[a,]
lab=as.factor(an[,"polygon_label"])



txt=paste(output,"/raw_data.RData",sep="")
save(r_tile,r_nuclei,a,an,lab,file=txt)
}

print(dim(r_tile))

xy=as.matrix(an[,c("x","y")])

xy=xy[a,]
nn=length(a)
top1=multi_SPARKX(r_tile,xy,as.factor(rep(1,length(a))),n.cores = 4)
r_tile=r_tile[,top1[1:100]]
top2=multi_SPARKX(r_nuclei,xy,as.factor(rep(1,length(a))),n.cores = 4)
r_nuclei=r_nuclei[,top2[1:100]]
data=cbind(r_tile,r_nuclei)

#xy=as.matrix(an[,c("x","y","area","eccentricity","solidity")])

#xy=xy[a,]
#xy=scale(xy)



data2=scale(data)
# Guard against low-sample runs and constant columns
data2[is.na(data2)] <- 0
max_nv <- min(50, nrow(data2) - 1, ncol(data2) - 1)
if (is.na(max_nv) || max_nv < 2) {
  stop(
    paste0(
      "Not enough observations/features for PCA after preprocessing: ",
      "nrow=", nrow(data2), ", ncol=", ncol(data2),
      ". Need at least 3 cells and 3 varying features."
    )
  )
}
pca_results <- irlba(A = data2, nv = max_nv)
pca <- pca_results$u %*% diag(pca_results$d)



txt=paste(output,"/raw_data_2.RData",sep="")
save(pca,a,an,lab,file=txt)



#lab=an[,"annotation"]
#lab[lab==""]=NA
#lab=as.factor(lab)

if (!dir.exists(output)) {
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
}


pca_dim <- ncol(pca)
txt=paste(output,"/pca_full_",pca_dim,".pdf",sep="")
pdf(txt)
plot(pca,pch=20,col=lab)
dev.off()

txt=paste(output,"/pca_full_",pca_dim,".RData",sep="")
save(pca,xy,file=txt)




u=umap(pca)$layout
rownames(u)=a
txt=paste(output,"/umap_full_",pca_dim,".pdf",sep="")
pdf(txt)
plot(u,pch=20,col=lab,cex=0.5)
dev.off()

rownames(u)=a
txt=paste(output,"/umap_full_",pca_dim,".RData",sep="")
save(u,xy,an,a,an,lab,file=txt)


dims_to_run <- 10 
jj <- KODAMA.matrix.parallel(pca[, 1: dims_to_run, drop = FALSE],
        spatial = spatial_for_kodama,
        landmarks = 1000,
        n.cores = 4,
        seed = 543210,
        ancestry = FALSE
      )

config <- umap.defaults
config$n_neighbors <- min(30, nrow(pca) - 1)
config$n_threads <- 4
vis <- KODAMA.visualization(jj, config = config)
    

rownames(vis) <- a
txt <- paste(output, "/kodama_full_", dims_to_run, ".pdf", sep = "")
pdf(txt)
plot(vis, pch = 20, col = lab, cex = 0.5)
dev.off()

txt <- paste(output, "/kodama_full_", dims_to_run, ".RData", sep = "")
save(vis, xy, a, an, lab, file = txt)
  


