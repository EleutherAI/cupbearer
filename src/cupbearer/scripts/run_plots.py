import argparse
import os
import pandas as pd
from cupbearer.scripts.data_loader import get_data, filters
from cupbearer.scripts.plotting import (
    barplot_by_dataset,
    plot_auc_roc_by_layer_by_score,
    logits_hist,
    plot_losses,
    create_tables,
    plot_scatter_variance,
    plot_scatter_logit,
    plot_all_trusted_test_label_balance,
    plot_score_correlations,
    plot_quirky_coefficient
)
from cupbearer.scripts.config import ONLINE_SCORE_ORDER, OFFLINE_SCORE_ORDER
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, required=True, 
                       choices=["barplot", "auc_layer", "scatter_variance", 
                                "scatter_logit", "logits_hist", "plot_losses", 
                                "create_tables", "label_balance", "label_balance_trusted_test", 
                                "label_balance_all", "correlations", "quirky_coefficient"], 
                       help="Select plot type")
    parser.add_argument("--multilayer", action="store_true", help="Toggle multilayer mode")
    parser.add_argument("--disagree", action="store_true", help="Toggle disagree mode")
    parser.add_argument("--type", type=str, default="online", 
                       choices=["online", "offline"], help="Type for create_tables")
    parser.add_argument("--prompt", type=str, default="Alice", help="Prompt for logits_hist")
    parser.add_argument("--labels", type=str, default="Alice", help="Labels for logits_hist")
    parser.add_argument("--loss", type=str, default="bob_logits", help="Loss field for plot_losses")
    parser.add_argument("--dataset", type=str, default="sciq", help="Dataset for quirky_lm task")
    parser.add_argument("--score1", type=str, help="First score for correlation plot")
    parser.add_argument("--score2", type=str, help="Second score for correlation plot")
    args = parser.parse_args()

    score_order = ONLINE_SCORE_ORDER if args.type == "online" else OFFLINE_SCORE_ORDER

    if args.plot == "barplot":
        df, _ = get_data(filters["rand_vs_nonrand"])
        barplot_by_dataset(df, compare="random_names", disagree=args.disagree)
    elif args.plot == "auc_layer":
        df, _ = get_data(filters["layerwise_agnostic"], score_order=score_order)
        df.loc[df["score"].isin(["rephrase"]), "layer"] = -1
        plot_auc_roc_by_layer_by_score(df, multilayer=args.multilayer, disagree=args.disagree, type=args.type)
    elif args.plot == "scatter_variance":
        df, _ = get_data(filters["layerwise_agnostic"])
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
    elif args.plot == "label_balance":
        plot_all_trusted_test_label_balance()
    elif args.plot == "correlations":
        df, _ = get_data(filters["layerwise_agnostic"])
        VALID_SCORES = df['score'].unique()
        if not args.score1 or not args.score2:
            raise ValueError(f"Must specify both --score1 and --score2 for correlation plots, valid scores are: {VALID_SCORES}")
        plot_score_correlations(df, args.score1, args.score2)
    elif args.plot == "quirky_coefficient":
        df, _ = get_data(filters["layerwise_agnostic"], include_all_datasets=True)
        plot_quirky_coefficient(df)

if __name__ == "__main__":
    main()