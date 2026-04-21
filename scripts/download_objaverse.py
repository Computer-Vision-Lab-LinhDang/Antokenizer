import objaverse
import objaverse.xl as oxl

annotations = oxl.get_annotations(
    download_dir="./.objaverse" # default download directory
)
annotations

# sample a single object from each source
sampled_df = annotations[(annotations["fileType"] == "glb") & (annotations["source"] == "sketchfab")].head(10).reset_index(drop=True)
sampled_df

oxl.download_objects(objects=sampled_df, download_dir="./.objaverse")

