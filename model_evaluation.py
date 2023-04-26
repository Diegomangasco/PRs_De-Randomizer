def clustering_error(cluster_labels: list, devices_number: int) -> float:
    single_clusters = len(set(cluster_labels))
    err = abs(devices_number - single_clusters) / devices_number
    err = 100 * err
    return err


def clustering_accuracy(cluster_labels: list, probe_sent_device_IDs: list, devices_number: int) -> None:
    device_labels = {key: list() for key in range(devices_number)}
    device_assignments = {key: dict() for key in range(devices_number)}

    for i, label in enumerate(cluster_labels):
        device_labels[probe_sent_device_IDs[i]].append(label)

    for k in device_labels.keys():
        seen = set()
        for label in device_labels[k]:
            if label not in seen:
                seen.add(label)
                device_assignments[k][label] = 1
            else:
                device_assignments[k][label] += 1

    print(device_assignments)

