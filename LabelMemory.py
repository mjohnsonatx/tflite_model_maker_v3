import json
import os
import threading

LABEL_MEMORY_FILENAME = "annotation_label_memory.json"
DEFAULT_LABEL = "kettlebell"


class LabelMemory:
    """Persistent shared store for annotation labels.

    Each tab creates its own LabelMemory instance. When one instance updates the
    labels, the others reload from disk and notify their UI listeners so every
    tab stays in sync.
    """

    _registry = []
    _registry_lock = threading.Lock()

    def __init__(self, app_root):
        self.app_root = app_root
        self.config_path = os.path.join(app_root, LABEL_MEMORY_FILENAME)
        self._labels = []
        self._current_label = ""
        self._listeners = []
        self._load()
        with LabelMemory._registry_lock:
            LabelMemory._registry.append(self)

    @property
    def labels(self):
        return list(self._labels)

    @property
    def current_label(self):
        return self._current_label

    def subscribe(self, callback):
        if callable(callback) and callback not in self._listeners:
            self._listeners.append(callback)

    def unsubscribe(self, callback):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def save(self, labels, current_label=None):
        cleaned = _clean_labels(labels)
        resolved_current = _normalize_current_label(current_label, cleaned, self._current_label)
        if resolved_current and resolved_current not in cleaned:
            cleaned.append(resolved_current)
        self._labels = cleaned
        self._current_label = resolved_current
        self._persist()
        self._broadcast()
        return list(self._labels)

    def merge_labels(self, new_labels, current_label=None):
        combined = list(self._labels)
        for label in _clean_labels(new_labels):
            if label not in combined:
                combined.append(label)
        resolved_current = _normalize_current_label(current_label, combined, self._current_label)
        if resolved_current and resolved_current not in combined:
            combined.append(resolved_current)
        self._labels = combined
        self._current_label = resolved_current
        self._persist()
        self._broadcast()
        return list(self._labels)

    def reload(self):
        previous = (list(self._labels), self._current_label)
        self._load()
        if previous != (list(self._labels), self._current_label):
            self._notify_listeners()

    def _load(self):
        if not os.path.exists(self.config_path):
            self._labels = [DEFAULT_LABEL]
            self._current_label = DEFAULT_LABEL
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            labels = _clean_labels(data.get("labels") or [])
            current = str(data.get("current_label") or "").strip()
            if not labels:
                labels = [DEFAULT_LABEL]
            if current not in labels:
                current = labels[0]
            self._labels = labels
            self._current_label = current
        except (OSError, ValueError, json.JSONDecodeError):
            self._labels = [DEFAULT_LABEL]
            self._current_label = DEFAULT_LABEL

    def _persist(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "labels": list(self._labels),
                        "current_label": self._current_label,
                    },
                    handle,
                    indent=2,
                )
        except OSError:
            pass

    def _notify_listeners(self):
        labels = self.labels
        current = self._current_label
        for listener in list(self._listeners):
            try:
                listener(labels, current)
            except Exception:
                pass

    def _broadcast(self):
        self._notify_listeners()
        with LabelMemory._registry_lock:
            peers = [
                instance for instance in LabelMemory._registry if instance is not self
            ]
        for peer in peers:
            try:
                peer.reload()
            except Exception:
                pass


def _clean_labels(labels):
    seen = []
    for label in labels or []:
        if label is None:
            continue
        text = str(label).strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def _normalize_current_label(requested, labels, fallback):
    if requested is not None:
        text = str(requested).strip()
        if text:
            return text
    if fallback and fallback in labels:
        return fallback
    if labels:
        return labels[0]
    return ""
