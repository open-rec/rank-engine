"""Convert LightGBM trees without losing split semantics or sigmoid precision."""

def convert_model(booster):
    from treelite.model_builder import (
        Metadata, ModelBuilder, PostProcessorFunc, TreeAnnotation,
    )

    dump = booster.dump_model()
    trees = dump["tree_info"]
    # A feature can have different missing-value semantics in different trees.
    # Add separate columns for Zero splits, leaving ordinary splits untouched.
    zero_features = set()
    pending = [tree["tree_structure"] for tree in trees]
    while pending:
        node = pending.pop()
        if "split_index" in node:
            if node["decision_type"] == "<=" and node["missing_type"] == "Zero":
                zero_features.add(node["split_feature"])
            pending.extend((node["left_child"], node["right_child"]))
        elif node.get("leaf_coeff"):
            raise ValueError("nvForest does not support LightGBM linear leaves")
    zero_features = sorted(zero_features)
    extra_columns = {feature: booster.num_feature() + i
                     for i, feature in enumerate(zero_features)}
    builder = ModelBuilder(
        threshold_type="float64", leaf_output_type="float64",
        metadata=Metadata(
            num_feature=booster.num_feature() + len(zero_features),
            task_type="kRegressor", average_tree_output=dump["average_output"],
            num_target=1, num_class=[1], leaf_vector_shape=(1, 1),
        ),
        tree_annotation=TreeAnnotation(len(trees), [0] * len(trees), [0] * len(trees)),
        # Treelite stores sigmoid_alpha as float32, even for double forests.
        # Return raw margins and apply the original double sigmoid separately.
        postprocessor=PostProcessorFunc("identity"), base_scores=[0.0],
    )
    for tree in trees:
        builder.start_tree()
        pending = [(0, tree["tree_structure"])]
        next_key = 1
        while pending:
            key, node = pending.pop()
            builder.start_node(key)
            if "split_index" not in node:
                builder.leaf(node["leaf_value"])
            else:
                left, right = next_key, next_key + 1
                next_key += 2
                feature = node["split_feature"]
                children = dict(left_child_key=left, right_child_key=right)
                if node["decision_type"] == "==":
                    builder.categorical_test(
                        feature, default_left=False,
                        category_list=[int(x) for x in node["threshold"].split("||")],
                        category_list_right_child=False, **children,
                    )
                elif node["decision_type"] == "<=":
                    missing_type = node["missing_type"]
                    default_left = node["default_left"]
                    if missing_type == "Zero":
                        feature = extra_columns[feature]
                    elif missing_type == "None":
                        default_left = 0.0 <= node["threshold"]
                    builder.numerical_test(
                        feature, node["threshold"], default_left=default_left,
                        opname="<=", **children,
                    )
                else:
                    raise ValueError(f"Unknown LightGBM split: {node['decision_type']}")
                pending.extend(((right, node["right_child"]), (left, node["left_child"])))
            builder.end_node()
        builder.end_tree()
    sigmoid = next(float(part.split(":", 1)[1])
                   for part in dump["objective"].split() if part.startswith("sigmoid:"))
    return builder.commit(), zero_features, sigmoid
