import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "ZcutBatchLoadImages";
const UPLOADED_IMAGES_WIDGET = "uploaded_images";
const UPLOAD_MODE_WIDGET = "upload_mode";
const LEGACY_PREVIEW_WIDGET = "uploaded_images_preview";
const FILE_LIST_WIDGET = "uploaded_file_names";
const FILE_LIST_EMPTY_HEIGHT = 40;
const FILE_LIST_PADDING_Y = 10;
const FILE_LIST_HEADER_HEIGHT = 22;
const FILE_LIST_ROW_HEIGHT = 22;
const MAX_UPLOAD_CONCURRENCY = 4;

function setDirty(node) {
    node.graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
}

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function setWidgetValue(node, widget, value) {
    if (!widget) {
        return;
    }

    widget.value = value;
    widget.callback?.(value, app.canvas, node, app.canvas?.graph_mouse, {});

    const index = node.widgets?.indexOf(widget) ?? -1;
    if (index >= 0 && Array.isArray(node.widgets_values)) {
        node.widgets_values[index] = value;
    }
}

function hideWidget(widget) {
    if (!widget) {
        return;
    }

    widget.hidden = true;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
}

function removeLegacyPreviewWidget(node) {
    if (!node.widgets?.length) {
        return;
    }

    const removed = node.widgets.filter((widget) => widget.name === LEGACY_PREVIEW_WIDGET);
    if (!removed.length) {
        return;
    }

    for (const widget of removed) {
        widget.onRemove?.();
        widget.element?.remove?.();
    }
    node.widgets = node.widgets.filter((widget) => widget.name !== LEGACY_PREVIEW_WIDGET);
}

function cleanupInternalWidgets(node) {
    removeLegacyPreviewWidget(node);
    hideWidget(getWidget(node, UPLOADED_IMAGES_WIDGET));
    updateFileListWidget(node);

    const size = node.computeSize?.();
    if (size) {
        node.setSize?.([node.size?.[0] || size[0], size[1]]);
    }
    setDirty(node);
}

function uploadedImageNames(node) {
    const uploadWidget = getWidget(node, UPLOADED_IMAGES_WIDGET);
    return parseUploadedImages(uploadWidget?.value)
        .map((item) => {
            if (typeof item === "string") {
                return item;
            }
            return item?.original_name || item?.name || "";
        })
        .filter(Boolean);
}

function uploadedImageRefs(node) {
    return parseUploadedImages(getWidget(node, UPLOADED_IMAGES_WIDGET)?.value);
}

function setUploadedImageRefs(node, refs) {
    setWidgetValue(node, getWidget(node, UPLOADED_IMAGES_WIDGET), JSON.stringify(refs));
    cleanupInternalWidgets(node);
}

function middleEllipsis(ctx, text, maxWidth) {
    if (ctx.measureText(text).width <= maxWidth) {
        return text;
    }

    const extensionMatch = text.match(/(\.[^.\\/]+)$/);
    const extension = extensionMatch?.[1] || "";
    const stem = extension ? text.slice(0, -extension.length) : text;
    const suffix = extension || stem.slice(-6);
    const suffixSource = extension ? stem : stem.slice(0, -6);
    let left = suffixSource;
    let right = suffix;

    while (left.length > 1 || right.length > suffix.length) {
        const candidate = `${left}...${right}`;
        if (ctx.measureText(candidate).width <= maxWidth) {
            return candidate;
        }
        if (left.length > 1) {
            left = left.slice(0, -1);
        } else {
            right = right.slice(1);
        }
    }

    return "...";
}

class UploadedFileNamesWidget {
    constructor(node) {
        this.name = FILE_LIST_WIDGET;
        this.type = "custom";
        this.node = node;
        this.value = "";
        this.serialize = false;
        this.names = [];
        this.hitAreas = [];
        this.height = FILE_LIST_EMPTY_HEIGHT;
    }

    refresh() {
        this.names = uploadedImageNames(this.node);
        const statusRows = this.node._zcutUploadStatus ? 1 : 0;
        this.height = this.names.length || statusRows
            ? FILE_LIST_PADDING_Y * 2 + FILE_LIST_HEADER_HEIGHT + (this.names.length + statusRows) * FILE_LIST_ROW_HEIGHT
            : FILE_LIST_EMPTY_HEIGHT;
    }

    computeSize(width) {
        this.refresh();
        return [width, this.height];
    }

    draw(ctx, node, widgetWidth, y, widgetHeight) {
        this.refresh();
        const names = this.names;
        this.hitAreas = [];
        const margin = 10;
        const x = margin;
        const width = widgetWidth - margin * 2;
        const top = y + 3;
        const height = Math.max(widgetHeight, this.height) - 6;
        const paddingX = 14;

        ctx.save();
        ctx.fillStyle = "#191a1f";
        ctx.strokeStyle = "#33363f";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(x, top, width, height, 8);
        ctx.fill();
        ctx.stroke();

        if (!names.length) {
            ctx.font = "12px Arial";
            ctx.fillStyle = "#777983";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(node._zcutUploadStatus || "Uploaded image names will appear here", x + paddingX, top + height / 2);
            ctx.restore();
            return;
        }

        this.widgetTopY = y;
        this.widgetWidth = widgetWidth;
        this.widgetHeight = widgetHeight;

        const headerY = top + FILE_LIST_PADDING_Y + FILE_LIST_HEADER_HEIGHT / 2;
        ctx.font = "12px Arial";
        ctx.fillStyle = "#9ca0aa";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(`${names.length} uploaded image${names.length === 1 ? "" : "s"}`, x + paddingX, headerY);

        const clearText = "clear all";
        ctx.font = "12px Arial";
        const clearWidth = Math.ceil(ctx.measureText(clearText).width) + 12;
        const clearX = x + width - paddingX - clearWidth;
        const clearY = headerY - 9;
        ctx.fillStyle = "#25262c";
        ctx.strokeStyle = "#41444d";
        ctx.beginPath();
        ctx.roundRect(clearX, clearY, clearWidth, 18, 5);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#c5c7d0";
        ctx.textAlign = "center";
        ctx.fillText(clearText, clearX + clearWidth / 2, headerY);
        this.hitAreas.push({ type: "clear", x: clearX, y: clearY, w: clearWidth, h: 18 });

        if (node._zcutUploadStatus) {
            ctx.font = "12px Arial";
            ctx.fillStyle = "#a1a1aa";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(node._zcutUploadStatus, x + paddingX, top + FILE_LIST_PADDING_Y + FILE_LIST_HEADER_HEIGHT + FILE_LIST_ROW_HEIGHT / 2);
        }

        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.font = "bold 12px Arial";
        const removeSize = 16;
        const textWidth = width - paddingX * 2 - removeSize - 8;
        for (let index = 0; index < names.length; index += 1) {
            const statusOffset = node._zcutUploadStatus ? FILE_LIST_ROW_HEIGHT : 0;
            const rowY = top + FILE_LIST_PADDING_Y + FILE_LIST_HEADER_HEIGHT + FILE_LIST_ROW_HEIGHT / 2 + statusOffset + index * FILE_LIST_ROW_HEIGHT;
            ctx.fillStyle = "#e4e4e7";
            ctx.fillText(middleEllipsis(ctx, names[index], textWidth), x + paddingX, rowY);

            const removeX = x + width - paddingX - removeSize;
            const removeY = rowY - removeSize / 2;
            ctx.fillStyle = "#2a2b31";
            ctx.strokeStyle = "#494b55";
            ctx.beginPath();
            ctx.roundRect(removeX, removeY, removeSize, removeSize, 4);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = "#c5c7d0";
            ctx.font = "bold 12px Arial";
            ctx.textAlign = "center";
            ctx.fillText("x", removeX + removeSize / 2, rowY + 0.5);
            ctx.textAlign = "left";
            ctx.font = "bold 12px Arial";
            this.hitAreas.push({ type: "remove", index, x: removeX, y: removeY, w: removeSize, h: removeSize });
        }

        ctx.restore();
    }

    mouse(event, pos, node) {
        if (event.type !== "pointerdown" && event.type !== "mousedown") {
            return false;
        }

        const [x, y] = pos;
        for (const area of this.hitAreas) {
            if (x < area.x || x > area.x + area.w || y < area.y || y > area.y + area.h) {
                continue;
            }

            const refs = uploadedImageRefs(node);
            if (area.type === "clear") {
                setUploadedImageRefs(node, []);
                return true;
            }

            if (area.type === "remove") {
                refs.splice(area.index, 1);
                setUploadedImageRefs(node, refs);
                return true;
            }
        }

        return false;
    }
}

function installFileList(node) {
    const existing = getWidget(node, FILE_LIST_WIDGET);
    if (existing) {
        existing.refresh?.();
        return existing;
    }

    const widget = new UploadedFileNamesWidget(node);
    widget.refresh();
    node.addCustomWidget(widget);
    return widget;
}

function updateFileListWidget(node) {
    getWidget(node, FILE_LIST_WIDGET)?.refresh?.();
}

function uploadFileName(file, index) {
    const dotIndex = file.name.lastIndexOf(".");
    const base = dotIndex > 0 ? file.name.slice(0, dotIndex) : file.name;
    const extension = dotIndex > 0 ? file.name.slice(dotIndex) : "";
    const safeBase = base.replace(/[\\/:*?"<>|]/g, "_").slice(0, 80) || "image";
    return `zcut_${Date.now()}_${index + 1}_${safeBase}${extension}`;
}

async function uploadImageFile(file, index) {
    const uploadFile = new File([file], uploadFileName(file, index), {
        type: file.type || "application/octet-stream",
        lastModified: file.lastModified,
    });
    const body = new FormData();
    body.append("image", uploadFile);
    body.append("type", "input");

    const response = await api.fetchApi("/upload/image", {
        method: "POST",
        body,
    });

    if (!response.ok) {
        throw new Error(`${file.name}: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    data.original_name = file.name;
    return data;
}

async function uploadImageFiles(files, onProgress) {
    const uploaded = [];
    const errors = [];
    let nextIndex = 0;
    let completed = 0;

    async function worker() {
        while (nextIndex < files.length) {
            const index = nextIndex;
            nextIndex += 1;
            try {
                uploaded[index] = await uploadImageFile(files[index], index);
            } catch (error) {
                errors.push(error.message || String(error));
            } finally {
                completed += 1;
                onProgress?.(completed, files.length);
            }
        }
    }

    const workers = Array.from({ length: Math.min(MAX_UPLOAD_CONCURRENCY, files.length) }, () => worker());
    await Promise.all(workers);

    return {
        uploaded: uploaded.filter(Boolean),
        errors,
    };
}

function parseUploadedImages(value) {
    try {
        const parsed = JSON.parse(value || "[]");
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
}

function installBatchUploadButton(node) {
    if (node._zcutBatchUploadInstalled) {
        return;
    }
    node._zcutBatchUploadInstalled = true;

    cleanupInternalWidgets(node);

    node.addWidget("button", "choose images", "Select Multiple Images", async () => {
        if (node._zcutUploading) {
            return;
        }

        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.multiple = true;

        input.onchange = async () => {
            const files = Array.from(input.files ?? []);
            if (!files.length) {
                return;
            }

            const uploadWidget = getWidget(node, UPLOADED_IMAGES_WIDGET);
            const modeWidget = getWidget(node, UPLOAD_MODE_WIDGET);
            node._zcutUploading = true;
            node._zcutUploadStatus = `Uploading 0/${files.length}...`;
            cleanupInternalWidgets(node);

            const { uploaded, errors } = await uploadImageFiles(files, (completed, total) => {
                node._zcutUploadStatus = `Uploading ${completed}/${total}...`;
                cleanupInternalWidgets(node);
            });

            node._zcutUploading = false;
            node._zcutUploadStatus = "";

            if (uploaded.length) {
                const existing = parseUploadedImages(uploadWidget?.value);
                setWidgetValue(node, uploadWidget, JSON.stringify([...existing, ...uploaded]));
                setWidgetValue(node, modeWidget, "multi_upload");
            }

            installFileList(node);
            cleanupInternalWidgets(node);
            setDirty(node);

            if (errors.length) {
                alert(`Some images failed to upload:\n${errors.join("\n")}`);
            }
        };

        input.click();
    });

    installFileList(node);
    cleanupInternalWidgets(node);
    setTimeout(() => cleanupInternalWidgets(node), 0);
    setTimeout(() => cleanupInternalWidgets(node), 250);
}

app.registerExtension({
    name: "Zcut.BatchLoadImages",
    nodeCreated(node) {
        if (node.constructor?.comfyClass !== NODE_CLASS) {
            return;
        }

        installBatchUploadButton(node);
    },
});
