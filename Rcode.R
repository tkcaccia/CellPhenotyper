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
for(ii in 1:length(ll)){
  if(endsWith(ll[ii], "csv.gz")){
    df <- fread(ll[ii])
    r_tile <- rbind(r_tile, df)
  }
}


ll=list.files( "embeddings_uni2_cyto",full.names=TRUE,recursive=TRUE)
r_nuclei=NULL
for(ii in 1:length(ll)){
  if(endsWith(ll[ii], "csv.gz")){
    df <- fread(ll[ii])
    r_nuclei <- rbind(r_nuclei, df)
  }
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
pca_results <- irlba(A = data2, nv = 50)
pca <- pca_results$u %*% diag(pca_results$d)



txt=paste(output,"/raw_data_2.RData",sep="")
save(pca,a,an,lab,file=txt)



#lab=an[,"annotation"]
#lab[lab==""]=NA
#lab=as.factor(lab)

if (!dir.exists(output)) {
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
}


txt=paste(output,"/pca_full_20.pdf",sep="")
pdf(txt)
plot(pca,pch=20,col=lab)
dev.off()

txt=paste(output,"/pca_full_20.RData",sep="")
save(pca,xy,file=txt)




u=umap(pca)$layout
rownames(u)=a
txt=paste(output,"/umap_full_20.pdf",sep="")
pdf(txt)
plot(u,pch=20,col=lab,cex=0.5)
dev.off()

rownames(u)=a
txt=paste(output,"/umap_full_20.RData",sep="")
save(u,xy,an,a,an,lab,file=txt)


if(TRUE){
	jj=KODAMA.matrix.parallel(pca[,1:10],
                          spatial = xy,
                          landmarks = 10000,
                          n.cores=8,
                          seed = 543210,
	                  ancestry=FALSE)

        config=umap.defaults
        config$n_neighbors=30
        config$n_threads = 8
        vis <- KODAMA.visualization(jj, config=config)


        rownames(vis)=a
        txt=paste(output,"/kodama_full_10.pdf",sep="")
        pdf(txt)
        plot(vis,pch=20,col=lab,cex=0.5)
        dev.off()

        txt=paste(output,"/kodama_full_10.RData",sep="")
        save(vis,xy,a,an,lab,file=txt)


        jj=KODAMA.matrix.parallel(pca[,1:20],
                          spatial = xy,
                          landmarks = 10000,
                          n.cores=8,
                          seed = 543210,
                          ancestry=FALSE)

        config=umap.defaults
        config$n_neighbors=30
        config$n_threads = 8
        vis <- KODAMA.visualization(jj, config=config)


        rownames(vis)=a
        txt=paste(output,"/kodama_full_20.pdf",sep="")
        pdf(txt)
        plot(vis,pch=20,col=lab,cex=0.5)
        dev.off()

        txt=paste(output,"/kodama_full_20.RData",sep="")
        save(vis,xy,a,an,lab,file=txt)


        jj=KODAMA.matrix.parallel(pca[,1:30],
                          spatial = xy,
                          landmarks = 10000,
                          n.cores=8,
                          seed = 543210,
                          ancestry=FALSE)

        config=umap.defaults
      #  config$n_neighbors=30
        config$n_threads = 8
        vis <- KODAMA.visualization(jj, config=config)


        rownames(vis)=a
        txt=paste(output,"/kodama_full_30.pdf",sep="")
        pdf(txt)
        plot(vis,pch=20,col=lab,cex=0.5)
        dev.off()

        txt=paste(output,"/kodama_full_30.RData",sep="")
        save(vis,xy,a,an,lab,file=txt)

        jj=KODAMA.matrix.parallel(pca,
                          spatial = xy,
                          landmarks = 10000,
                          n.cores=8,
                          seed = 543210,
                          ancestry=FALSE)

        config=umap.defaults
      #  config$n_neighbors=30
        config$n_threads = 8
        vis <- KODAMA.visualization(jj, config=config)


        rownames(vis)=a
        txt=paste(output,"/kodama_full_50.pdf",sep="")
        pdf(txt)
        plot(vis,pch=20,col=lab,cex=0.5)
        dev.off()

        txt=paste(output,"/kodama_full_50.RData",sep="")
        save(vis,xy,a,an,lab,file=txt)


}






