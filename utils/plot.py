import matplotlib.pyplot as plt


def plot_interp_acc(lambdas, train_acc_interp_naive, test_acc_interp_naive,
                    train_acc_interp_clever, test_acc_interp_clever,
                    train_acc_interp_ensemble=None, test_acc_interp_ensemble=None,
                    test_acc_data_cond=None, test_acc_connect=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lambdas,
            train_acc_interp_naive,
            linestyle="dashed",
            color="tab:blue",
            alpha=0.5,
            linewidth=2,
            label="Train, naïve interp.")
    ax.plot(lambdas,
            test_acc_interp_naive,
            linestyle="dashed",
            color="tab:orange",
            alpha=0.5,
            linewidth=2,
            label="Test, naïve interp.")
    ax.plot(lambdas,
            train_acc_interp_clever,
            linestyle="solid",
            color="tab:blue",
            linewidth=2,
            label="Train, permuted interp.")
    ax.plot(lambdas,
            test_acc_interp_clever,
            linestyle="solid",
            color="tab:orange",
            linewidth=2,
            label="Test, permuted interp.")
    if train_acc_interp_ensemble is not None:
        ax.plot(lambdas,
                train_acc_interp_ensemble,
                linestyle="dashdot",
                color="tab:blue",
                linewidth=2,
                label="Train, ensemble interp.")
        ax.plot(lambdas,
                test_acc_interp_ensemble,
                linestyle="dashdot",
                color="tab:orange",
                linewidth=2,
                label="Test, ensemble interp.")
    if test_acc_data_cond is not None:
        ax.plot(lambdas,
                [test_acc_data_cond for _ in lambdas],
                linestyle="dotted",
                color="tab:orange",
                linewidth=2,
                label="Test, data condensation.")
    if test_acc_connect is not None:
        ax.plot(lambdas,
                [test_acc_connect for _ in lambdas],
                linestyle="dotted",
                color="tab:green",
                linewidth=2,
                label="Test, connection.")
    ax.set_xlabel("$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Accuracy")
    # TODO label x=0 tick as \theta_1, and x=1 tick as \theta_2
    ax.set_title(f"Accuracy between the two models")
    ax.legend(loc="lower right", framealpha=0.5)
    fig.tight_layout()
    return fig


def plot_interp_loss(lambdas, train_acc_interp_naive, test_acc_interp_naive,
                     train_acc_interp_clever, test_acc_interp_clever,
                     train_acc_interp_ensemble=None, test_acc_interp_ensemble=None,
                     test_loss_data_cond=None, test_loss_connect=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lambdas,
            train_acc_interp_naive,
            linestyle="dashed",
            color="tab:blue",
            alpha=0.5,
            linewidth=2,
            label="Train, naïve interp.")
    ax.plot(lambdas,
            test_acc_interp_naive,
            linestyle="dashed",
            color="tab:orange",
            alpha=0.5,
            linewidth=2,
            label="Test, naïve interp.")
    ax.plot(lambdas,
            train_acc_interp_clever,
            linestyle="solid",
            color="tab:blue",
            linewidth=2,
            label="Train, permuted interp.")
    ax.plot(lambdas,
            test_acc_interp_clever,
            linestyle="solid",
            color="tab:orange",
            linewidth=2,
            label="Test, permuted interp.")
    if train_acc_interp_ensemble is not None:
        ax.plot(lambdas,
                train_acc_interp_ensemble,
                linestyle="dashdot",
                color="tab:blue",
                linewidth=2,
                label="Train, ensemble interp.")
        ax.plot(lambdas,
                test_acc_interp_ensemble,
                linestyle="dashdot",
                color="tab:orange",
                linewidth=2,
                label="Test, ensemble interp.")
    if test_loss_data_cond is not None:
        ax.plot(lambdas,
                [test_loss_data_cond for _ in lambdas],
                linestyle="dotted",
                color="tab:orange",
                linewidth=2,
                label="Test, data condensation.")
    if test_loss_connect is not None:
        ax.plot(lambdas,
                [test_loss_connect for _ in lambdas],
                linestyle="dotted",
                color="tab:green",
                linewidth=2,
                label="Test, connection.")

    ax.set_xlabel("$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Loss")
    # TODO label x=0 tick as \theta_1, and x=1 tick as \theta_2
    ax.set_title(f"Loss between the two models")
    ax.legend(loc="lower right", framealpha=0.5)
    fig.tight_layout()
    return fig
