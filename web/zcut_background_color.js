import { app } from "../../scripts/app.js";

const NODE_CLASS = "ZcutAddBackground";
const WIDGET_NAME = "background_color";
const DEFAULT_COLOR = "#ffffff";

function normalizeColor(value) {
    const text = String(value ?? "").trim();
    const shortMatch = text.match(/^#?([0-9a-fA-F]{3})$/);
    if (shortMatch) {
        return `#${shortMatch[1]
            .split("")
            .map((channel) => channel + channel)
            .join("")
            .toLowerCase()}`;
    }

    const fullMatch = text.match(/^#?([0-9a-fA-F]{6})$/);
    if (fullMatch) {
        return `#${fullMatch[1].toLowerCase()}`;
    }

    if (Number.isInteger(value)) {
        return `#${(value & 0xffffff).toString(16).padStart(6, "0")}`;
    }

    return DEFAULT_COLOR;
}

function setDirty(node) {
    node.graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
}

function syncBackgroundColor(node, sourceWidget, value) {
    if (node._zcutSyncingBackgroundColor) {
        return;
    }

    node._zcutSyncingBackgroundColor = true;
    try {
        const color = normalizeColor(value);
        for (const widget of node.widgets ?? []) {
            if (widget.name !== WIDGET_NAME) {
                continue;
            }
            widget.value = color;
            widget.callback?.(color, app.canvas, node, app.canvas?.graph_mouse, {});
        }

        const index = node.widgets?.findIndex((widget) => widget.name === WIDGET_NAME) ?? -1;
        if (index >= 0 && Array.isArray(node.widgets_values)) {
            node.widgets_values[index] = color;
        }
        setDirty(node);
    } finally {
        node._zcutSyncingBackgroundColor = false;
    }
}

function hookWidget(node, widget) {
    if (widget._zcutBackgroundColorSynced) {
        return;
    }
    widget._zcutBackgroundColorSynced = true;

    const originalCallback = widget.callback;
    widget.callback = function (value, ...args) {
        const result = originalCallback?.call(this, value, ...args);
        syncBackgroundColor(node, widget, this.value ?? value);
        return result;
    };

    let storedValue = widget.value;
    try {
        Object.defineProperty(widget, "value", {
            configurable: true,
            enumerable: true,
            get() {
                return storedValue;
            },
            set(value) {
                storedValue = normalizeColor(value);
                if (!node._zcutSyncingBackgroundColor) {
                    queueMicrotask(() => syncBackgroundColor(node, widget, storedValue));
                }
            },
        });
        widget.value = storedValue;
    } catch {
        widget.value = normalizeColor(storedValue);
    }
}

function installBackgroundColorSync(node) {
    if (node._zcutBackgroundColorSyncInstalled) {
        return;
    }

    const widgets = node.widgets?.filter((widget) => widget.name === WIDGET_NAME) ?? [];
    if (!widgets.length) {
        return;
    }

    node._zcutBackgroundColorSyncInstalled = true;
    for (const widget of widgets) {
        hookWidget(node, widget);
    }

    syncBackgroundColor(node, widgets[0], widgets[0].value);
}

app.registerExtension({
    name: "Zcut.BackgroundColorSync",
    nodeCreated(node) {
        if (node.constructor?.comfyClass !== NODE_CLASS) {
            return;
        }

        installBackgroundColorSync(node);
    },
});
