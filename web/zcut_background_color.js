import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_CLASS = "ZcutAddBackground";
const WIDGET_NAME = "background_color";
const PICKER_WIDGET_NAME = "__zcut_background_color_picker";
const DEFAULT_COLOR = "#ffffff";

let colorInput = null;

function getColorInput() {
    if (colorInput) {
        return colorInput;
    }

    colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.style.position = "fixed";
    colorInput.style.left = "-1000px";
    colorInput.style.top = "-1000px";
    colorInput.style.width = "1px";
    colorInput.style.height = "1px";
    colorInput.style.opacity = "0";
    colorInput.style.pointerEvents = "none";
    document.body.appendChild(colorInput);
    return colorInput;
}

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

    return DEFAULT_COLOR;
}

function roundedRect(ctx, x, y, width, height, radius) {
    if (ctx.roundRect) {
        ctx.roundRect(x, y, width, height, radius);
        return;
    }

    const r = Math.min(radius, width / 2, height / 2);
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
}

function drawSwatch(ctx, node, widget) {
    const widgetHeight = LiteGraph.NODE_WIDGET_HEIGHT || 20;
    const rowY = widget.last_y ?? 0;
    const swatchWidth = 72;
    const swatchHeight = Math.max(16, widgetHeight - 6);
    const x = Math.max(0, node.size[0] - swatchWidth - 24);
    const y = rowY + Math.max(2, (widgetHeight - swatchHeight) / 2);
    const radius = swatchHeight / 2;
    const color = normalizeColor(widget.value);

    widget._zcutColorSwatchRect = [x, y, swatchWidth, swatchHeight];

    ctx.save();
    ctx.beginPath();
    roundedRect(ctx, x, y, swatchWidth, swatchHeight, radius);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(0, 0, 0, 0.28)";
    ctx.stroke();
    ctx.restore();
}

function isInsideRect(pos, rect) {
    return rect && pos[0] >= rect[0] && pos[0] <= rect[0] + rect[2] && pos[1] >= rect[1] && pos[1] <= rect[1] + rect[3];
}

function getPickerRoot(widget) {
    return widget?.parentEl ?? widget?.element ?? widget?.inputEl ?? null;
}

function getPickerButton(widget) {
    const root = getPickerRoot(widget);
    if (!root) {
        return null;
    }

    if (root instanceof HTMLButtonElement) {
        return root;
    }

    return root.querySelector?.("button") ?? root;
}

function hidePickerWidget(widget) {
    widget.type = "converted-widget";
    widget.hidden = true;
    widget.serialize = false;
    widget.computeSize = () => [0, -4];

    const root = getPickerRoot(widget);
    if (root) {
        Object.assign(root.style, {
            position: "fixed",
            left: "-10000px",
            top: "-10000px",
            width: "1px",
            height: "1px",
            minWidth: "1px",
            minHeight: "1px",
            opacity: "0",
            pointerEvents: "none",
            zIndex: "1800",
        });
    }
}

function syncWidgetValue(node, widget, value) {
    const normalized = normalizeColor(value);
    widget.value = normalized;

    const widgetIndex = node.widgets?.indexOf?.(widget) ?? -1;
    if (widgetIndex >= 0 && Array.isArray(node.widgets_values)) {
        node.widgets_values[widgetIndex] = normalized;
    }

    widget.callback?.(normalized, app.canvas, node, app.canvas?.graph_mouse, {});
    node.graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
    return normalized;
}

function ensureBuiltinPickerWidget(node, widget) {
    if (node._zcutBuiltinColorPickerWidget) {
        return node._zcutBuiltinColorPickerWidget;
    }

    const createColorWidget = ComfyWidgets?.COLOR;
    if (typeof createColorWidget !== "function") {
        return null;
    }

    const result = createColorWidget(node, PICKER_WIDGET_NAME, ["COLOR", { default: normalizeColor(widget.value), format: "hex" }], app);
    const pickerWidget = result?.widget ?? result;
    if (!pickerWidget) {
        return null;
    }

    const originalCallback = pickerWidget.callback;
    pickerWidget.callback = function (value, ...args) {
        const normalized = normalizeColor(value);
        pickerWidget.value = normalized;
        syncWidgetValue(node, widget, normalized);
        return originalCallback?.call(this, normalized, ...args);
    };

    hidePickerWidget(pickerWidget);
    node._zcutBuiltinColorPickerWidget = pickerWidget;
    return pickerWidget;
}

function positionBuiltinPicker(node, widget, pickerWidget) {
    const rect = widget._zcutColorSwatchRect;
    const canvas = app.canvas?.canvas;
    const root = getPickerRoot(pickerWidget);
    if (!rect || !canvas || !root) {
        return;
    }

    const scale = app.canvas?.ds?.scale ?? 1;
    const offset = app.canvas?.ds?.offset ?? [0, 0];
    const canvasRect = canvas.getBoundingClientRect();
    const left = canvasRect.left + (node.pos[0] + rect[0]) * scale;
    const top = canvasRect.top + (node.pos[1] + rect[1]) * scale;
    const width = Math.max(1, rect[2] * scale);
    const height = Math.max(1, rect[3] * scale);

    Object.assign(root.style, {
        position: "fixed",
        left: `${left}px`,
        top: `${top}px`,
        width: `${width}px`,
        height: `${height}px`,
        minWidth: `${width}px`,
        minHeight: `${height}px`,
        opacity: "0.001",
        pointerEvents: "none",
        zIndex: "1800",
    });

    const button = getPickerButton(pickerWidget);
    if (button) {
        Object.assign(button.style, {
            width: "100%",
            height: "100%",
            pointerEvents: "auto",
        });
    }
}

function openColorPicker(node, widget) {
    const pickerWidget = ensureBuiltinPickerWidget(node, widget);
    if (pickerWidget) {
        const normalized = normalizeColor(widget.value);
        pickerWidget.value = normalized;
        positionBuiltinPicker(node, widget, pickerWidget);
        const button = getPickerButton(pickerWidget);
        button?.click?.();
        window.setTimeout(() => hidePickerWidget(pickerWidget), 800);
        return;
    }

    const input = getColorInput();
    input.value = normalizeColor(widget.value);

    input.oninput = () => {
        syncWidgetValue(node, widget, input.value);
    };

    input.onchange = input.oninput;
    input.click();
}

function installBackgroundColorSwatch(node) {
    if (node._zcutBackgroundColorSwatchInstalled) {
        return;
    }

    const widget = node.widgets?.find((item) => item.name === WIDGET_NAME);
    if (!widget) {
        return;
    }

    node._zcutBackgroundColorSwatchInstalled = true;

    const originalDraw = widget.draw;
    widget.draw = function (ctx) {
        originalDraw?.apply(this, arguments);
        drawSwatch(ctx, node, this);
    };

    const originalOnMouseDown = node.onMouseDown;
    node.onMouseDown = function (pos) {
        if (isInsideRect(pos, widget._zcutColorSwatchRect)) {
            openColorPicker(this, widget);
            return true;
        }

        return originalOnMouseDown?.apply(this, arguments);
    };
}

app.registerExtension({
    name: "Zcut.BackgroundColorSwatch",
    nodeCreated(node) {
        if (node.constructor?.comfyClass !== NODE_CLASS) {
            return;
        }

        installBackgroundColorSwatch(node);
    },
});
