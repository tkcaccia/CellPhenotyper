

singularity exec --bind "$PWD":"$PWD" --pwd "$PWD" ${SINGULARITY} \
  Rscript bin/Rcode_Clustering.R \
  output/KODAMA \
  >> logs/KODAMA_${BASE}.Rout


echo "R code COMPLETED SUCCESSFULLY"


# merges islands within ~2*radius px gaps
singularity exec --bind "$PWD":"$PWD" --pwd "$PWD" "${SINGULARITY}" \
python bin/labels_to_cluster_mask.py \
  --mask out_stardist_roi/labels_cyto.tif \
  --map output/KODAMA/cluster.csv \
  --out output/KODAMA/cluster_mask.tif \
  --preview output/KODAMA/cluster_mask_preview.png \
  --preview-factor 10





# merges islands within ~2*radius px gaps
singularity exec --bind "$PWD":"$PWD" --pwd "$PWD" "${SINGULARITY}" \
python bin/grow_to_tissue.py \
  --image out_stardist_roi/crop_roi.tif \
  --mask output/KODAMA/cluster_mask.tif \
  --tissue-mask output/KODAMA/tissue_mask.tif \
  --out output/KODAMA/grown_mask.ome.tif \
  --preview output/KODAMA/qc_preview_10x.png \
  --preview-factor 10 \
  --restrict-to-seeded-components \
  --min-seed-area 200 \
  --fill-holes-area 50000 \
  --close-radius 12 \
  --nuclei-thresh 170 \
  --nuclei-dilate 2 \
  --pyr-compression LZW \
  --max-workers 16 \
  --downsample GAUSSIAN \
  --overwrite



# merges islands within ~2*radius px gaps
# 3) Labeled mask (clusters / grown mask) -> one feature per region, label stored in properties.value
singularity exec --bind "$PWD":"$PWD" --pwd "$PWD" "${SINGULARITY}" \
python bin/mask_to_geojson.py \
  --mask output/KODAMA/grown_mask.ome.tif \
  --page 0 \
  --out output/KODAMA/grown_mask_smooth_class.geojson \
  --dissolve-by-value \
  --min-area 500 \
  --smooth-buffer 10.0 \
  --smooth-passes 3 \
  --simplify 6.0 \
  --fill-holes \
  --preserve-topology \
  --group-prefix "color_"



echo "PIPELINE COMPLETED SUCCESSFULLY"

