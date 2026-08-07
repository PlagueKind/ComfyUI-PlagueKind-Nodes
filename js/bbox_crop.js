import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const ASPECT_PRESETS = {
    "1:1": 1 / 1,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "21:9": 21 / 9,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
};

const PREVIEW_HEIGHT = 260;
const HANDLE = 8;

function getRatio(node) {
    const modeW = node.widgets.find(w => w.name === "aspect_ratio");
    if (!modeW || modeW.value === "Free") return null;
    if (modeW.value === "Custom") {
        const cw = node.widgets.find(w => w.name === "custom_ratio_w")?.value || 1;
        const ch = node.widgets.find(w => w.name === "custom_ratio_h")?.value || 1;
        return cw / Math.max(1, ch);
    }
    return ASPECT_PRESETS[modeW.value] ?? null;
}

function getBoxWidgets(node) {
    return {
        x: node.widgets.find(w => w.name === "crop_x"),
        y: node.widgets.find(w => w.name === "crop_y"),
        w: node.widgets.find(w => w.name === "crop_w"),
        h: node.widgets.find(w => w.name === "crop_h"),
    };
}

function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
}

app.registerExtension({
    name: "VisualBBoxCrop.UI",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VisualBBoxCrop") {
            return;
        }

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

            const node = this;
            const cropValueWidgets = ["crop_x", "crop_y", "crop_w", "crop_h"];

            function applyVisibility() {
                const toggle = node.widgets?.find(w => w.name === "show_crop_values");

                for (const w of node.widgets || []) {
                    if (cropValueWidgets.includes(w.name)) {
                        w.hidden = toggle ? !toggle.value : true;
                    }
                }

                const computed = node.computeSize();
                node.setSize([node.size[0], computed[1]]);

                if (node.graph) {
                    node.graph.setDirtyCanvas(true, true);
                }
            }

            const toggleWidget = node.widgets?.find(w => w.name === "show_crop_values");
            if (toggleWidget) {
                const origCallback = toggleWidget.callback;
                toggleWidget.callback = function (...args) {
                    if (origCallback) origCallback.apply(this, args);
                    applyVisibility();
                };
            }

            requestAnimationFrame(() => applyVisibility());

            node._cropImg = null;
            node._imgRect = null; // {x, y, w, h} in widget-local canvas space
            node._dragMode = null; // "move" | "tl" | "tr" | "bl" | "br" | null

            const widget = node.addWidget("crop_canvas", "crop_preview", "", () => {}, { serialize: false });

            widget.computeSize = function () {
                return [0, node._cropImg ? PREVIEW_HEIGHT : 26];
            };

            widget.draw = function (ctx, node, widget_width, y, widget_height) {
                ctx.save();

                if (!node._cropImg) {
                    ctx.fillStyle = "rgba(255,255,255,0.15)";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText("Run once to load a preview", widget_width * 0.5, y + 17);
                    ctx.restore();
                    return;
                }

                const img = node._cropImg;
                const scale = Math.min(widget_width / img.naturalWidth, PREVIEW_HEIGHT / img.naturalHeight);
                const iw = img.naturalWidth * scale;
                const ih = img.naturalHeight * scale;
                const ix = (widget_width - iw) * 0.5;
                const iy = y;

                node._imgRect = { x: ix, y: iy, w: iw, h: ih };

                ctx.drawImage(img, ix, iy, iw, ih);

                const { x, y: wy, w, h } = getBoxWidgets(node);
                const bx = ix + (x?.value ?? 0) * iw;
                const by = iy + (wy?.value ?? 0) * ih;
                const bw = (w?.value ?? 1) * iw;
                const bh = (h?.value ?? 1) * ih;

                ctx.fillStyle = "rgba(0,0,0,0.55)";
                ctx.fillRect(ix, iy, iw, by - iy);
                ctx.fillRect(ix, by + bh, iw, iy + ih - (by + bh));
                ctx.fillRect(ix, by, bx - ix, bh);
                ctx.fillRect(bx + bw, by, ix + iw - (bx + bw), bh);

                ctx.strokeStyle = "#5ecbff";
                ctx.lineWidth = 2;
                ctx.strokeRect(bx, by, bw, bh);

                ctx.fillStyle = "#5ecbff";
                for (const [hx, hy] of [[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]]) {
                    ctx.fillRect(hx - HANDLE / 2, hy - HANDLE / 2, HANDLE, HANDLE);
                }

                ctx.restore();
            };

            widget.mouse = function (event, pos, node) {
                if (!node._cropImg || !node._imgRect) return false;

                const rect = node._imgRect;
                const { x, y, w, h } = getBoxWidgets(node);
                if (!x || !y || !w || !h) return false;

                const bx = rect.x + x.value * rect.w;
                const by = rect.y + y.value * rect.h;
                const bw = w.value * rect.w;
                const bh = h.value * rect.h;

                const [mx, my] = pos;

                if (event.type === "pointerdown" || event.type === "mousedown") {
                    const corners = {
                        tl: [bx, by], tr: [bx + bw, by],
                        bl: [bx, by + bh], br: [bx + bw, by + bh],
                    };
                    node._dragMode = null;
                    for (const [name, [cx, cy]] of Object.entries(corners)) {
                        if (Math.abs(mx - cx) <= HANDLE && Math.abs(my - cy) <= HANDLE) {
                            node._dragMode = name;
                            break;
                        }
                    }
                    if (!node._dragMode && mx >= bx && mx <= bx + bw && my >= by && my <= by + bh) {
                        node._dragMode = "move";
                        node._dragOffset = [mx - bx, my - by];
                    }
                    return !!node._dragMode;
                }

                if ((event.type === "pointermove" || event.type === "mousemove") && node._dragMode) {
                    const ratio = getRatio(node);
                    let nx = bx, ny = by, nw = bw, nh = bh;

                    if (node._dragMode === "move") {
                        nx = clamp(mx - node._dragOffset[0], rect.x, rect.x + rect.w - bw);
                        ny = clamp(my - node._dragOffset[1], rect.y, rect.y + rect.h - bh);
                    } else {
                        const opp = {
                            tl: [bx + bw, by + bh], tr: [bx, by + bh],
                            bl: [bx + bw, by], br: [bx, by],
                        }[node._dragMode];

                        let px2 = clamp(mx, rect.x, rect.x + rect.w);
                        let py2 = clamp(my, rect.y, rect.y + rect.h);

                        if (ratio) {
                            const dx = px2 - opp[0];
                            const dy = py2 - opp[1];
                            const signX = dx < 0 ? -1 : 1;
                            const signY = dy < 0 ? -1 : 1;
                            const targetW = Math.max(4, Math.abs(dx));
                            const targetH = targetW / ratio;
                            px2 = opp[0] + signX * targetW;
                            py2 = opp[1] + signY * targetH;
                        }

                        nx = Math.min(opp[0], px2);
                        ny = Math.min(opp[1], py2);
                        nw = Math.max(4, Math.abs(px2 - opp[0]));
                        nh = Math.max(4, Math.abs(py2 - opp[1]));

                        nw = Math.min(nw, rect.x + rect.w - nx);
                        nh = Math.min(nh, rect.y + rect.h - ny);
                    }

                    x.value = clamp((nx - rect.x) / rect.w, 0, 1);
                    y.value = clamp((ny - rect.y) / rect.h, 0, 1);
                    w.value = clamp(nw / rect.w, 0.001, 1 - x.value);
                    h.value = clamp(nh / rect.h, 0.001, 1 - y.value);

                    node.graph?.setDirtyCanvas(true, true);
                    return true;
                }

                if (event.type === "pointerup" || event.type === "mouseup") {
                    const had = !!node._dragMode;
                    node._dragMode = null;
                    return had;
                }

                return false;
            };

            return r;
        };

        const origOnExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onExecuted = function (message) {
            const r = origOnExecuted ? origOnExecuted.apply(this, arguments) : undefined;
            const node = this;

            const imgs = message?.pk_crop_preview;
            if (imgs && imgs.length) {
                const info = imgs[0];
                const url = api.apiURL(
                    `/view?filename=${encodeURIComponent(info.filename)}&subfolder=${encodeURIComponent(info.subfolder || "")}&type=${info.type || "temp"}`
                );
                const im = new Image();
                im.onload = () => {
                    node._cropImg = im;
                    const w = node.widgets?.find(w => w.name === "crop_preview");
                    if (w) {
                        const computed = node.computeSize();
                        node.setSize([node.size[0], computed[1]]);
                    }
                    node.graph?.setDirtyCanvas(true, true);
                };
                im.src = url;
            }

            return r;
        };
    }
});
