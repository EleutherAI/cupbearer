import argparse
import os
from cupbearer.scripts.data_loader import get_data, filters
from cupbearer.scripts.plotting import (
    barplot_by_dataset,
    plot_auc_roc_by_layer_by_score,
    logits_hist,
    plot_losses,
    create_tables,
    plot_scatter_variance,
    plot_scatter_logit
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, required=True, 
                       choices=["barplot", "auc_layer", "scatter_variance", 
                               "scatter_logit", "logits_hist", "plot_losses", 
                               "create_tables"], 
                       help="Select plot type")
    parser.add_argument("--multilayer", action="store_true", help="Toggle multilayer mode")
    parser.add_argument("--disagree", action="store_true", help="Toggle disagree mode")
    parser.add_argument("--type", type=str, default="online", 
                       choices=["online", "offline"], help="Type for create_tables")
    parser.add_argument("--prompt", type=str, default="Alice", help="Prompt for logits_hist")
    parser.add_argument("--labels", type=str, default="Alice", help="Labels for logits_hist")
    parser.add_argument("--loss", type=str, default="bob_logits", help="Loss field for plot_losses")
    args = parser.parse_args()

    if args.plot == "barplot":
        df, _ = get_data(filters["rand_vs_nonrand"])
        df = df[~df["dataset"].isin(["modularaddition", "squaring"])]
        barplot_by_dataset(df, compare="random_names", disagree=args.disagree)
    elif args.plot == "auc_layer":
        df, _ = get_data(filters["layerwise_agnostic"])
        df.loc[df["score"].isin(["rephrase"]), "layer"] = -1
        plot_auc_roc_by_layer_by_score(df, multilayer=args.multilayer, disagree=args.disagree)
    elif args.plot == "scatter_variance":
        df, _ = get_data(filters["none"], log_dir=os.path.join("..", "logs", "adv_image"))
        plot_scatter_variance(df)
    elif args.plot == "scatter_logit":
        df, _ = get_data(filters["none"], log_dir=os.path.join("..", "logs", "adv_image"))
        plot_scatter_logit(df)
    elif args.plot == "logits_hist":
        logits_hist(args.prompt, args.labels)
    elif args.plot == "plot_losses":
        plot_losses(args.loss, f"{args.loss} by Base Model", args.loss)
    elif args.plot == "create_tables":
        create_tables(args.type)

if __name__ == "__main__":
    main()