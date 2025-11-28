# Define hyperparameter values for grid search
method="doscond"
lr_feat_values=(1e-4 1e-3 1e-2)
lr_adj_values=(1e-4 1e-3 1e-2)
dis_metric_values=("mse" "ours")
outer_loop=1
inner_loop=1
threshold_values=(0.0)


for lr_feat in "${lr_feat_values[@]}"; do
  for lr_adj in "${lr_adj_values[@]}"; do
    for dis_metric in "${dis_metric_values[@]}"; do
      for threshold in "${threshold_values[@]}"; do
          CUDA_VISIBLE_DEVICES="$gpu_id" python train_all.py -M "$method" -D "$dataset"\
              --mode grid_search \
              --condense_model "$condense_model" \
              --lr_feat "$lr_feat" \
              --lr_adj "$lr_adj" \
              --dis_metric "$dis_metric" \
              --outer_loop "$outer_loop" \
              --inner_loop "$inner_loop" \
              --threshold "$threshold" 
      done
    done
  done
done

