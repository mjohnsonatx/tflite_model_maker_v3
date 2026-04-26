import json
import os


class LabelMemory:
    DEFAULT_LABELS = ["kettlebell"]
    FILE_NAME = "annotation_label_memory.json"

    def __init__(self, app_root):
        self.path = os.path.join(app_root, self.FILE_NAME)
        self.labels = list(self.DEFAULT_LABELS)
        self.current_label = self.labels[0]
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return

        try:
            with open(self.path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return

        labels = self._normalize_labels(data.get('labels', []))
        current_label = str(data.get('current_label', '')).strip()

        if labels:
            self.labels = labels

        if current_label:
            self.current_label = current_label
            if current_label not in self.labels:
                self.labels.insert(0, current_label)
        elif self.current_label not in self.labels:
            self.current_label = self.labels[0]

    def save(self, labels=None, current_label=None):
        if labels is not None:
            self.labels = self._normalize_labels(labels)

        if current_label is not None:
            current_label = str(current_label).strip()
            if current_label:
                self.current_label = current_label

        if not self.labels:
            self.labels = list(self.DEFAULT_LABELS)

        if current_label is not None and not current_label:
            self.current_label = self.labels[0]
        elif self.current_label not in self.labels:
            self.labels.insert(0, self.current_label)

        data = {
            'labels': self.labels,
            'current_label': self.current_label,
        }

        temp_path = f"{self.path}.tmp"
        with open(temp_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)
            file.write('\n')
        os.replace(temp_path, self.path)

    def merge_labels(self, labels, current_label=None):
        merged = list(self.labels)
        for label in self._normalize_labels(labels):
            if label not in merged:
                merged.append(label)

        self.save(merged, current_label=current_label)
        return list(self.labels)

    def remove_labels(self, labels):
        labels_to_remove = set(self._normalize_labels(labels))
        if not labels_to_remove:
            return list(self.labels)

        remaining = [
            label for label in self.labels
            if label not in labels_to_remove
        ]

        if not remaining:
            remaining = list(self.DEFAULT_LABELS)

        current_label = self.current_label
        if current_label in labels_to_remove or current_label not in remaining:
            current_label = remaining[0]

        self.save(remaining, current_label=current_label)
        return list(self.labels)

    @staticmethod
    def _normalize_labels(labels):
        normalized = []
        seen = set()

        for label in labels:
            label = str(label).strip()
            if not label or label in seen:
                continue
            normalized.append(label)
            seen.add(label)

        return normalized
