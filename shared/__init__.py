"""
Shared utilities for the Intention-Recognition framework.
"""

from .data_utils import (
    load_node_coords,
    load_time_series,
    generate_time_labels,
    build_correlation_graph,
    build_hybrid_graph,
    TrafficIntentDataset,
    compute_future_labels,
)
from .models import (
    GATLayer,
    SpatialEncoder,
    BiGRUTemporal,
    TemporalConvNet,
    AttentionPooling,
    IntentModel,
)
from .training import train_model
from .visualization import (
    save_figure,
    plot_correlation_graph,
    plot_loss_curves,
    visualize_intent_tsne,
    plot_confusion_matrix,
    plot_predictions,
)
from .metrics import (
    compute_metrics,
    compute_persistence_metrics,
    collect_predictions,
)
