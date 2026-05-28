import { app } from "/scripts/app.js";

app.registerExtension({
    name: "UnifiedResizeImageMask.UI",

    async beforeRegisterNodeDef(nodeType, nodeData) {

        if (nodeData.name !== "UnifiedResizeImageMask") {
            return;
        }

        const LABELS = {
            scale_mode: "Scale Mode",
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
                      "Multiplier": ["multiplier"],
                      "Total Pixels (MP)": ["megapixels"],
                      "Longer Side": ["long_side_target"],
                      "Shorter Side": ["short_side_target"],
        };

        const dimensionWidgets = [
            "width",
            "height",
            "multiplier",
            "megapixels",
            "long_side_target",
            "short_side_target"
        ];

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {

            const r = origOnNodeCreated
            ? origOnNodeCreated.apply(this, arguments)
            : undefined;

            const node = this;

            function applyVisibility() {

                const modeWidget = node.widgets?.find(
                    w => w.name === "scale_mode"
                );

                if (!modeWidget) {
                    return;
                }

                const show = modeMap[modeWidget.value] || [];

                for (const w of node.widgets || []) {

                    if (LABELS[w.name]) {
                        w.label = LABELS[w.name];
                    }

                    if (dimensionWidgets.includes(w.name)) {
                        w.hidden = !show.includes(w.name);
                    }
                }

                node.setSize(node.computeSize());

                if (node.graph) {
                    node.graph.setDirtyCanvas(true, true);
                }
            }

            const modeWidget = node.widgets?.find(
                w => w.name === "scale_mode"
            );

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

            return r;
        };
    }
});
