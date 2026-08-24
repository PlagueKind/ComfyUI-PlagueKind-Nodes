import { app } from "/scripts/app.js";

app.registerExtension({
    name: "UnifiedResizeImageMask.UI",

    setup() {
        const applyPrototypePatch = () => {
            for (const node of app.graph?._nodes ?? []) {
                for (const w of node.widgets ?? []) {
                    if (w.constructor?.name === "PromotedWidgetView" && !w.constructor.prototype._resolutionLabelPatched) {
                        const proto = w.constructor.prototype;
                        proto._resolutionLabelPatched = true;

                        const origDraw = proto.draw;
                        proto.draw = function(ctx, node, widget_width, y, widget_height, showText) {
                            const deep = this.resolveDeepest?.();
                            if (deep?.widget?.isResolutionLabel) {
                                ctx.save();

                                ctx.beginPath();
                                ctx.moveTo(15, y + 4);
                                ctx.lineTo(widget_width - 15, y + 4);
                                ctx.lineWidth = 1;
                                ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
                                ctx.stroke();

                                ctx.fillStyle = LiteGraph?.WIDGET_TEXT_COLOR ?? "#a9a9a9";
                                ctx.font = "13px Arial";
                                ctx.textAlign = "center";
                                ctx.fillText(this.value, widget_width * 0.5, y + 22);

                                ctx.restore();
                                return;
                            }
                            return origDraw.call(this, ctx, node, widget_width, y, widget_height, showText);
                        };
                        return true;
                    }
                }
            }
            return false;
        };

        const observer = new MutationObserver(() => {
            if (applyPrototypePatch()) observer.disconnect();
        });
            observer.observe(document.body, { childList: true, subtree: true });

            const interval = setInterval(() => {
                if (applyPrototypePatch()) clearInterval(interval);
            }, 500);

                setTimeout(() => clearInterval(interval), 30000);
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "UnifiedResizeImageMask") {
            return;
        }

        const LABELS = {
            scale_mode: "Scale Mode",
            aspect_ratio: "Aspect Ratio",
            width: "Width",
            height: "Height",
            multiplier: "Multiplier",
            megapixels: "Megapixels",
            upscale_method: "Scale Method",
            long_side_target: "Long Side Target",
            short_side_target: "Short Side Target",
            maintain_aspect: "Maintain Aspect",
            crop: "Crop",
            divisible_by: "Divisible By",
        };

        const modeMap = {
            "Dimensions (W × H)": ["width", "height"],
                      "Multiplier": ["multiplier", "aspect_ratio"],
                      "Total Pixels (MP)": ["megapixels", "aspect_ratio"],
                      "Longer Side": ["long_side_target", "aspect_ratio"],
                      "Shorter Side": ["short_side_target", "aspect_ratio"],
        };

        const dimensionWidgets = [
            "width",
            "height",
            "multiplier",
            "megapixels",
            "long_side_target",
            "short_side_target",
            "aspect_ratio"
        ];

        // Dimension widgets that should display as whole integers in the UI
        const intDisplayWidgets = [
            "width",
            "height",
            "long_side_target",
            "short_side_target",
            "divisible_by"
        ];

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated
            ? origOnNodeCreated.apply(this, arguments)
            : undefined;

            const node = this;

            function applyVisibility() {
                if (!node._all_inputs && node.inputs) {
                    node._all_inputs = [...node.inputs];
                }

                const modeWidget = node.widgets?.find(w => w.name === "scale_mode");

                if (!modeWidget) {
                    return;
                }

                const show = modeMap[modeWidget.value] || [];

                // 1. Configure widget labels, formatting, and visibility
                for (const w of node.widgets || []) {
                    if (LABELS[w.name]) {
                        w.label = LABELS[w.name];
                    }

                    // Forces precision to 0 so integers display without decimals (.0)
                    if (intDisplayWidgets.includes(w.name)) {
                        w.options = w.options || {};
                        w.options.precision = 0;
                    }

                    if (dimensionWidgets.includes(w.name)) {
                        w.hidden = !show.includes(w.name);
                    }
                }

                // 2. Rebuild node.inputs so only active sockets exist in the graph
                if (node._all_inputs) {
                    for (let i = node.inputs.length - 1; i >= 0; i--) {
                        const inp = node.inputs[i];
                        if (dimensionWidgets.includes(inp.name) && !show.includes(inp.name)) {
                            if (inp.link !== null) {
                                node.disconnectInput(i);
                            }
                        }
                    }

                    node.inputs = node._all_inputs.filter(inp => {
                        if (dimensionWidgets.includes(inp.name)) {
                            return show.includes(inp.name);
                        }
                        return true;
                    });
                }

                const computed = node.computeSize();
                node.setSize([node.size[0], computed[1]]);

                if (node.graph) {
                    node.graph.setDirtyCanvas(true, true);
                }
            }

            const modeWidget = node.widgets?.find(w => w.name === "scale_mode");

            if (modeWidget) {
                const origCallback = modeWidget.callback;

                modeWidget.callback = function (...args) {
                    if (origCallback) {
                        origCallback.apply(this, args);
                    }
                    applyVisibility();
                };
            }

            requestAnimationFrame(() => {
                applyVisibility();
            });

            const resWidget = node.addWidget("text", "resolution_display", "Resolution: (pending)", () => {}, { serialize: false });

            resWidget.mouse = () => false;
            resWidget.isResolutionLabel = true;

            resWidget.draw = function(ctx, node, widget_width, y, widget_height) {
                ctx.save();

                ctx.beginPath();
                ctx.moveTo(15, y + 4);
                ctx.lineTo(widget_width - 15, y + 4);
                ctx.lineWidth = 1;
                ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
                ctx.stroke();

                ctx.fillStyle = LiteGraph?.WIDGET_TEXT_COLOR ?? "#a9a9a9";
                ctx.font = "13px Arial";
                ctx.textAlign = "center";
                ctx.fillText(this.value, widget_width * 0.5, y + 22);

                ctx.restore();
            };

            resWidget.computeSize = function() {
                return [0, 30];
            };

            return r;
        };

        const origOnExecuted = nodeType.prototype.onExecuted;

        nodeType.prototype.onExecuted = function (message) {
            const r = origOnExecuted ? origOnExecuted.apply(this, arguments) : undefined;

            if (message?.text) {
                let widget = this.widgets?.find(w => w.name === "resolution_display");
                if (widget) {
                    widget.value = message.text[0];
                    const computed = this.computeSize();
                    this.setSize([this.size[0], computed[1]]);
                    if (this.graph) {
                        this.graph.setDirtyCanvas(true, true);
                    }
                }
            }

            return r;
        };
    }
});
