from collections import Counter


# Classes with zero accuracy
zero_acc_classes = [cls for cls, acc in per_class_acc.items() if acc == 0]
print("Classes with 0% accuracy and their most frequent wrong predictions:")

for zero_cls in zero_acc_classes:
    cls_name = imagenet_idx_to_class[zero_cls]
    wrong_list = wrong_preds[zero_cls]
    if not wrong_list:
        print(f"{cls_name}: No wrong predictions found (very strange).")
        continue

    # Count most frequent predicted wrong classes
    pred_classes = [pred for _, pred in wrong_list]
    pred_counts = Counter(pred_classes).most_common(5)  # top 5 most predicted wrong classes

    print(f"\n{cls_name} (true class):")
    for pred_cls_idx, count in pred_counts:
        pred_cls_name = imagenet_idx_to_class.get(pred_cls_idx, f"ImageNet class {pred_cls_idx}")
        print(f"  Predicted as {pred_cls_name} ({count} times)")
