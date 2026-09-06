import { app } from "../../scripts/app.js";

const NODE_TYPE = "H3SLAAttention";

// Standard ComfyUI widget-hiding trick (as used by e.g. rgthree-comfy):
// collapsing computeSize to zero height and tagging the type removes the
// widget from the node's drawn layout while leaving it in node.widgets, so
// its value still serializes with the workflow -- unlike splicing it out of
// node.widgets, which would shift the positional widgets_values indices of
// every legacy saved workflow that follows it.
const HIDDEN_TAG = "H3SLAHidden";

function hideWidget(node, widget) {
    if (widget.type?.startsWith(HIDDEN_TAG)) return;
    widget.origType = widget.type;
    widget.origComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
    widget.type = HIDDEN_TAG;
}

app.registerExtension({
    name: "PlagueKind.H3SLAAttention.ReferenceProtection",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            const node = this;

            function applyReferenceLabel() {
                const referenceMode = node.widgets?.find(
                    widget => widget.name === "reference_protection"
                );
                const protectAudio = node.widgets?.find(
                    widget => widget.name === "protect_audio"
                );
                const engineWidget = node.widgets?.find(
                    widget => widget.name === "engine"
                );
                const int8Pv = node.widgets?.find(
                    widget => widget.name === "use_int8_pv"
                );
                if (referenceMode) {
                    referenceMode.label = "Protect Image/Video Reference";
                    const legacyValue = String(referenceMode.value).toLowerCase();
                    if (legacyValue === "manual") {
                        referenceMode.value = "Light";
                    } else if (legacyValue === "true") {
                        // Pre-rename saved workflows stored the raw combo
                        // value "True"; patch.py's resolver still accepts it
                        // on the backend, but the frontend dropdown itself
                        // no longer has that option, so the widget would
                        // otherwise show a stale/invalid value on load.
                        referenceMode.value = "Heavy Enforcement";
                    }
                }
                if (protectAudio && typeof protectAudio.value === "string") {
                    const disabled = ["off", "false", "0", "no"].includes(
                        protectAudio.value.toLowerCase()
                    );
                    protectAudio.value = !disabled;
                }
                if (engineWidget) {
                    // "hybrid" was removed from the combo entirely; patch.py's
                    // resolver still accepts a saved "hybrid" value and maps
                    // it to "triton", but the dropdown no longer offers it,
                    // so re-point stale saved workflows the same way here.
                    if (String(engineWidget.value).toLowerCase() === "hybrid") {
                        engineWidget.value = "triton";
                    }
                }
                if (int8Pv) {
                    // Hidden pending a fix for the graph breaks it can cause
                    // under ComfyUI's newer Comfy Compiler -- force disabled
                    // regardless of what a saved workflow has stored.
                    int8Pv.value = false;
                    hideWidget(node, int8Pv);
                }
                node.graph?.setDirtyCanvas(true, true);
            }

            requestAnimationFrame(applyReferenceLabel);
            return result;
        };
    },
});
