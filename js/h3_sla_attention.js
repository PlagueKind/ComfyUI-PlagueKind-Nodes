import { app } from "../../scripts/app.js";

const NODE_TYPE = "H3SLAAttention";

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
                node.graph?.setDirtyCanvas(true, true);
            }

            requestAnimationFrame(applyReferenceLabel);
            return result;
        };
    },
});
